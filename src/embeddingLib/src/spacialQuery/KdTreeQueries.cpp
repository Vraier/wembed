#include "KdTreeQueries.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "Macros.hpp"

namespace {
// subtrees with fewer points are built sequentially (not worth a task)
constexpr uint32_t PARALLEL_BUILD_CUTOFF = 4096;
// number of points inspected to find the widest dimension of a node
constexpr uint32_t SPREAD_SAMPLE_SIZE = 64;
}  // namespace

KdTreeQueries::KdTreeQueries(const std::vector<CVecRef>& points, const size_t dimension) : dimension(dimension) {
    ASSERT(dimension >= 1);
    ASSERT(points.size() < std::numeric_limits<uint32_t>::max());
    ASSERT(std::all_of(points.begin(), points.end(), [&](CVecRef p) { return p.dimension() == dimension; }));
    numPoints = static_cast<uint32_t>(points.size());
    if (numPoints == 0) return;

    const size_t n = numPoints;
    const size_t d = dimension;
    std::vector<float> source(n * d);  // input order, row per point
    std::vector<BuildEntry> entries(n);
    float maxAbs = 0.0f;
#pragma omp parallel for default(none) firstprivate(n, d) shared(points, source, entries) reduction(max : maxAbs) \
    schedule(static)
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < d; k++) {
            const float value = static_cast<float>(points[i][k]);
            source[i * d + k] = value;
            maxAbs = std::max(maxAbs, std::abs(value));
        }
        entries[i] = {0.0f, static_cast<uint32_t>(i)};
    }
    maxAbsCoordinate = maxAbs;

    // all nodes of a level have (up to rounding) the same size, so the tree is complete
    // down to the last level of inner nodes
    size_t numInnerNodes = 0;
    size_t nodesInLevel = 1;
    for (size_t largest = n; largest > LEAF_SIZE; largest = (largest + 1) / 2) {
        numInnerNodes += nodesInLevel;
        nodesInLevel *= 2;
    }
    nodes.resize(numInnerNodes);
    coordinates.resize(n * d);
    ids.resize(n);

    const float* sourceData = source.data();
    BuildEntry* entryData = entries.data();
#pragma omp parallel default(none) firstprivate(sourceData, entryData) if (numPoints > PARALLEL_BUILD_CUTOFF)
#pragma omp single nowait
    build(0, 0, numPoints, sourceData, entryData);
}

void KdTreeQueries::build(const size_t node, const uint32_t lo, const uint32_t hi, const float* source,
                          BuildEntry* entries) {
    const uint32_t size = hi - lo;
    if (size <= LEAF_SIZE) {
        writeLeaf(lo, hi, source, entries);
        return;
    }

    const uint32_t dim = widestDimension(lo, hi, source, entries);
    for (uint32_t i = lo; i < hi; i++) {
        entries[i].key = source[static_cast<size_t>(entries[i].id) * dimension + dim];
    }
    const uint32_t mid = lo + size / 2;
    std::nth_element(entries + lo, entries + mid, entries + hi,
                     [](const BuildEntry& a, const BuildEntry& b) { return a.key < b.key; });
    ASSERT(node < nodes.size());
    nodes[node] = {entries[mid].key, dim};

    // the subtrees work on disjoint ranges of entries, nodes and coordinates
    if (size > PARALLEL_BUILD_CUTOFF) {
#pragma omp task default(none) firstprivate(node, lo, mid, source, entries)
        build(2 * node + 1, lo, mid, source, entries);
    } else {
        build(2 * node + 1, lo, mid, source, entries);
    }
    build(2 * node + 2, mid, hi, source, entries);
}

uint32_t KdTreeQueries::widestDimension(const uint32_t lo, const uint32_t hi, const float* source,
                                        const BuildEntry* entries) const {
    // The exact spread needs a pass over all points of the node on every level. A strided
    // sample finds the same dimension unless two spreads are close, where it does not matter.
    const uint32_t stride = std::max<uint32_t>(1, (hi - lo) / SPREAD_SAMPLE_SIZE);
    uint32_t widest = 0;
    float widestSpread = -1.0f;
    for (uint32_t k = 0; k < dimension; k++) {
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();
        for (uint32_t i = lo; i < hi; i += stride) {
            const float value = source[static_cast<size_t>(entries[i].id) * dimension + k];
            min = std::min(min, value);
            max = std::max(max, value);
        }
        if (max - min > widestSpread) {
            widestSpread = max - min;
            widest = k;
        }
    }
    return widest;
}

void KdTreeQueries::writeLeaf(const uint32_t lo, const uint32_t hi, const float* source, const BuildEntry* entries) {
    const size_t count = hi - lo;
    float* block = coordinates.data() + static_cast<size_t>(lo) * dimension;
    for (size_t j = 0; j < count; j++) {
        const uint32_t id = entries[lo + j].id;
        ids[lo + j] = id;
        for (size_t k = 0; k < dimension; k++) {
            block[k * count + j] = source[static_cast<size_t>(id) * dimension + k];
        }
    }
}

size_t KdTreeQueries::query_sphere(CVecRef point, const double radius, std::vector<uint64_t>& out) const {
    ASSERT(point.dimension() == dimension);
    ASSERT(radius >= 0.0);
    out.clear();
    if (numPoints == 0) return 0;

    thread_local std::vector<float> buffer;
    buffer.resize(2 * dimension);
    float* queryPoint = buffer.data();
    float* offsets = queryPoint + dimension;
    float maxAbs = maxAbsCoordinate;
    for (size_t k = 0; k < dimension; k++) {
        queryPoint[k] = static_cast<float>(point[k]);
        offsets[k] = 0.0f;  // the root cell is the whole space
        maxAbs = std::max(maxAbs, std::abs(queryPoint[k]));
    }

    // Pad the radius so that float32 rounding cannot drop a point that is within `radius`
    // in double precision. Converting a coordinate to float32 moves it by at most
    // |x| * 2^-24, for both the point and the query in each of the d dimensions (absolute
    // term). Evaluating the squared distance and the incremental cell distances adds a
    // relative error of roughly (d + tree depth) * 2^-24 (relative term, with headroom).
    const double d = static_cast<double>(dimension);
    const double absolutePad = 2.0 * std::sqrt(d) * static_cast<double>(maxAbs) * 0x1p-24;
    const double relativePad = (d + 64.0) * 0x1p-23;
    const double paddedRadius = (radius + absolutePad) * (1.0 + relativePad);

    Query query{queryPoint, offsets, static_cast<float>(paddedRadius * paddedRadius), &out};
    search(0, 0, numPoints, 0.0f, query);
    return out.size();
}

void KdTreeQueries::search(const size_t node, const uint32_t lo, const uint32_t hi, const float sqDistToCell,
                           Query& query) const {
    const uint32_t size = hi - lo;
    if (size <= LEAF_SIZE) {
        scanLeaf(lo, hi, query);
        return;
    }

    const Node inner = nodes[node];
    const uint32_t mid = lo + size / 2;
    const float diff = query.point[inner.dim] - inner.split;

    // the near child contains the query (w.r.t. this split), its cell distance is unchanged
    if (diff < 0.0f) {
        search(2 * node + 1, lo, mid, sqDistToCell, query);
    } else {
        search(2 * node + 2, mid, hi, sqDistToCell, query);
    }

    // the far child lies behind the split plane: along inner.dim its cell is |diff| away,
    // which replaces the distance the current cell had in that dimension
    const float oldOffset = query.offsets[inner.dim];
    const float farSqDist = sqDistToCell - oldOffset * oldOffset + diff * diff;
    if (farSqDist > query.sqRadius) return;

    query.offsets[inner.dim] = diff;
    if (diff < 0.0f) {
        search(2 * node + 2, mid, hi, farSqDist, query);
    } else {
        search(2 * node + 1, lo, mid, farSqDist, query);
    }
    query.offsets[inner.dim] = oldOffset;
}

void KdTreeQueries::scanLeaf(const uint32_t lo, const uint32_t hi, const Query& query) const {
    const uint32_t count = hi - lo;
    const float* block = coordinates.data() + static_cast<size_t>(lo) * dimension;

    // squared distances of all points in the leaf at once, one dimension after the other
    float sqDists[LEAF_SIZE];
    {
        const float q = query.point[0];
#pragma omp simd
        for (uint32_t j = 0; j < count; j++) {
            const float delta = block[j] - q;
            sqDists[j] = delta * delta;
        }
    }
    for (size_t k = 1; k < dimension; k++) {
        const float* column = block + k * count;
        const float q = query.point[k];
#pragma omp simd
        for (uint32_t j = 0; j < count; j++) {
            const float delta = column[j] - q;
            sqDists[j] += delta * delta;
        }
    }

    for (uint32_t j = 0; j < count; j++) {
        if (sqDists[j] <= query.sqRadius) {
            query.out->push_back(ids[lo + j]);
        }
    }
}
