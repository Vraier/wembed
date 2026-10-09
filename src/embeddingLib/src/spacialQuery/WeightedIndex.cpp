#include "WeightedIndex.hpp"

#include <limits>
#include <stdexcept>

#include "KdTreeQueries.hpp"
#ifdef WEMBED_HAS_SPRK
#include "SprkQueries.hpp"
#endif
#include "VectorOperations.hpp"

IndexType WeightedIndex::checkedIndexType(const IndexType type, const int dimension) {
    if (type == IndexType::Sprk) {
#ifndef WEMBED_HAS_SPRK
        throw std::invalid_argument(
            "wembed was built without the sprk tree (WEMBED_USE_SPRK=OFF). "
            "Select the slower KD-tree explicitly (index type 0).");
#endif
        if (dimension < 2 || dimension > 16) {
            throw std::invalid_argument("The sprk tree supports 2 to 16 dimensions, got " +
                                        std::to_string(dimension) +
                                        ". Select the slower KD-tree explicitly (index type 0).");
        }
    }
    return type;
}

void WeightedIndex::update(const VecList<>& newPositions, const std::vector<flt_t>& newWeights,
                           flt_t maxDisplacement) {
    ASSERT(newPositions.size() == newWeights.size(), "Positions and weights must have the same size");
    ASSERT(newPositions.dimension() == DIMENSION, "Positions must have the same dimension as the index");
    if (newWeights.size() != invExpWeights.size()) {
        maxDisplacement = std::numeric_limits<flt_t>::infinity();
    }
    this->positions = &newPositions;
    this->weights = &newWeights;
    updateCalls++;

    if (dynamicBuffer > 0.0) {
        // both endpoints of a pair move, so the pair distance changes by <= 2 * maxDisplacement
        remainingBudget -= 2.0 * maxDisplacement;
        if (mode != QueryMode::Plain && remainingBudget >= 0.0) {
            mode = QueryMode::Reuse;
            return;
        }
    }

    rebuildClasses();
    rebuildCalls++;

    if (dynamicBuffer > 0.0 && 2.0 * maxDisplacement * minExpectedReuses <= dynamicBuffer) {
        mode = QueryMode::Fill;
        remainingBudget = dynamicBuffer;
        cachedPairs.resize(newWeights.size());
    } else {
        mode = QueryMode::Plain;
        remainingBudget = -1.0;
    }
}

void WeightedIndex::rebuildClasses() {
    const std::vector<flt_t>& newWeights = *weights;
    const VecList<>& newPositions = *positions;

    invExpWeights.resize(newWeights.size());
#pragma omp parallel for default(none) shared(newWeights) schedule(static)
    for (size_t i = 0; i < newWeights.size(); i++) {
        invExpWeights[i] = flt_t{1.0} / Toolkit::myPow(newWeights[i], flt_t{1.0} / static_cast<flt_t>(DIMENSION));
    }

    const flt_t minWeight = *std::min_element(newWeights.begin(), newWeights.end());
    flt_t maxWeight = *std::max_element(newWeights.begin(), newWeights.end());
    classBounds.clear();
    for (flt_t bound = minWeight * doublingFactor; bound < maxWeight; bound *= doublingFactor) {
        classBounds.push_back(bound);
    }

    std::vector<std::vector<CVecRef>> classContent(classBounds.size() + 1);
    classToGlobal.assign(classBounds.size() + 1, {});
    for (NodeId v = 0; v < newPositions.size(); v++) {
        const auto c = std::upper_bound(classBounds.begin(), classBounds.end(), newWeights[v]) - classBounds.begin();
        classContent[c].push_back(newPositions[v]);
        classToGlobal[c].push_back(v);
    }

    maxWeightOfClass = classBounds;
    maxWeightOfClass.push_back(maxWeight);

    spacialIndices.clear();
    for (size_t i = 0; i < classContent.size(); i++) {
        switch (indexType) {
            case IndexType::KdTree:
                spacialIndices.push_back(std::make_unique<KdTreeQueries>(classContent[i], DIMENSION));
                break;
#ifdef WEMBED_HAS_SPRK
            case IndexType::Sprk:
                spacialIndices.push_back(std::make_unique<SprkQueries>(classContent[i], DIMENSION));
                break;
#endif
            default:
                throw std::logic_error("unknown index type");
        }
    }
}

void WeightedIndex::getOwnedRepellingPairs(const NodeId v, std::vector<NodeId>& out) {
    ASSERT(positions != nullptr, "update() must run before queries");
    out.clear();
    const flt_t weight = (*weights)[v];
    const flt_t invV = invExpWeights[v];

    if (mode == QueryMode::Reuse) {
        std::vector<NodeId>& cache = cachedPairs[v];
        size_t keep = 0;
        for (const NodeId u : cache) {
            const flt_t dist = vectorOperations::calculateLPNorm((*positions)[v], (*positions)[u]);
            const flt_t weightedDist = dist * invV * invExpWeights[u];
            // a pair beyond threshold + budget cannot come back below the threshold
            // before the next rebuild, so it is dropped for good
            if (weightedDist >= 1.0 + remainingBudget * invV * invExpWeights[u]) continue;
            cache[keep++] = u;
            if (weightedDist < 1.0) out.push_back(u);
        }
        cache.resize(keep);
        return;
    }

    thread_local std::vector<NodeId> candidates;
    candidates.clear();
    const auto ownClass = std::upper_bound(classBounds.begin(), classBounds.end(), weight) - classBounds.begin();
    const flt_t radiusSlack = mode == QueryMode::Fill ? dynamicBuffer : 0.0;
    for (size_t i = 0; i <= static_cast<size_t>(ownClass) && i < spacialIndices.size(); i++) {
        queryClass(i, (*positions)[v], weight, radiusSlack, candidates);
    }

    if (mode == QueryMode::Fill) {
        std::vector<NodeId>& cache = cachedPairs[v];
        cache.clear();
        for (const NodeId u : candidates) {
            if (u == v || !ownsPair(weight, (*weights)[u], v, u)) continue;
            const flt_t dist = vectorOperations::calculateLPNorm((*positions)[v], (*positions)[u]);
            const flt_t weightedDist = dist * invV * invExpWeights[u];
            // the class query radius over-covers light partners; keep only what can
            // reach the threshold within the buffer
            if (weightedDist >= flt_t{1.0} + dynamicBuffer * invV * invExpWeights[u]) continue;
            cache.push_back(u);
            if (weightedDist < flt_t{1.0}) out.push_back(u);
        }
        return;
    }

    // mode == QueryMode::Plain
    for (const NodeId u : candidates) {
        if (u == v || !ownsPair(weight, (*weights)[u], v, u)) continue;
        // pairs at weighted distance >= 1 contribute zero force and zero loss
        const flt_t dist = vectorOperations::calculateLPNorm((*positions)[v], (*positions)[u]);
        if (dist * invV * invExpWeights[u] >= flt_t{1.0}) continue;
        out.push_back(u);
    }
}

void WeightedIndex::queryClass(const size_t weightClass, CVecRef p, const flt_t weight, const flt_t radiusSlack,
                               std::vector<NodeId>& output) const {
    ASSERT(spacialIndices.size() == maxWeightOfClass.size(), "Indices and weight classes must have the same size");
    ASSERT(weightClass < maxWeightOfClass.size());

    const flt_t queryRadius =
        Toolkit::myPow(weight * maxWeightOfClass[weightClass], flt_t{1.0} / static_cast<flt_t>(DIMENSION)) + radiusSlack;
    ASSERT(queryRadius > 0);

    thread_local std::vector<uint64_t> localIds;
    spacialIndices[weightClass]->query_sphere(p, queryRadius, localIds);
    for (const uint64_t localId : localIds) {
        output.push_back(classToGlobal[weightClass][localId]);
    }
}
