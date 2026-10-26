#pragma once

#include <cstdint>
#include <vector>

#include "SpacialIndex.hpp"

/**
 * Plain C++ KD-tree for fixed-radius queries. Fallback for the sprk tree (SprkQueries)
 * when no rescent Rust is available at build time.
 *
 * Uses some of the SPRK-tree optimizations:
 *  - static and balanced: every node splits its points at the median position, 
 *    an inner node only stores split dimension and value
 *    (8 bytes per node, no child pointers)
 *  - points are stored in float32 and physically reordered so that every leaf is one
 *    contiguous block
 *  - split dimension = widest spread, estimated on a small sample of the node's points
 *  - pruning with incrementally updated squared distances
 *  - the build is parallelized with OpenMP tasks; queries are read-only and lock-free
 *
 * Distances are evaluated in float32. The query radius is slightly inflated,
 * so no point within `radius` is missed;
 */
class KdTreeQueries : public SpatialIndex {
   public:
    KdTreeQueries(const std::vector<CVecRef>& points, size_t dimension);

    size_t query_sphere(CVecRef point, double radius, std::vector<uint64_t>& out) const override;

    // largest number of points in a leaf; leaves hold between LEAF_SIZE / 2 and LEAF_SIZE points
    static constexpr uint32_t LEAF_SIZE = 32;

   private:
    struct Node {
        float split;
        uint32_t dim;
    };
    struct BuildEntry {
        float key;
        uint32_t id;
    };
    struct Query {
        const float* point;
        float* offsets;  // per dimension: distance of the query to the current cell
        float sqRadius;
        std::vector<uint64_t>* out;
    };

    void build(size_t node, uint32_t lo, uint32_t hi, const float* source, BuildEntry* entries);
    uint32_t widestDimension(uint32_t lo, uint32_t hi, const float* source, const BuildEntry* entries) const;
    void writeLeaf(uint32_t lo, uint32_t hi, const float* source, const BuildEntry* entries);

    void search(size_t node, uint32_t lo, uint32_t hi, float sqDistToCell, Query& query) const;
    void scanLeaf(uint32_t lo, uint32_t hi, const Query& query) const;

    size_t dimension;
    uint32_t numPoints = 0;
    float maxAbsCoordinate = 0.0f;  // for the rounding error bound

    // inner nodes in heap order: children of i are 2i+1 and 2i+2. Node i covers the slot
    // range [lo, hi) and splits it at mid = lo + (hi - lo) / 2 into [lo, mid) and [mid, hi)
    std::vector<Node> nodes;
    // leaf [lo, hi) owns coordinates[lo * d, hi * d); within the block, the values of
    // dimension k are stored contiguously at offset k * (hi - lo)
    std::vector<float> coordinates;
    std::vector<uint32_t> ids;  // slot -> position in the input point array
};
