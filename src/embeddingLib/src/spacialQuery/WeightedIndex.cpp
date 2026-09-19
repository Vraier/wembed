#include "WeightedIndex.hpp"

#include <limits>

#include "SprkQueries.hpp"
#include "VectorOperations.hpp"

void WeightedIndex::update(const VecList& newPositions, const std::vector<double>& newWeights,
                           double maxDisplacement) {
    ASSERT(newPositions.size() == newWeights.size(), "Positions and weights must have the same size");
    ASSERT(newPositions.dimension() == DIMENSION, "Positions must have the same dimension as the index");
    if (newWeights.size() != invExpWeights.size()) {
        maxDisplacement = std::numeric_limits<double>::infinity();
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

    if (dynamicBuffer > 0.0 && 2.0 * maxDisplacement * MIN_EXPECTED_REUSES <= dynamicBuffer) {
        mode = QueryMode::Fill;
        remainingBudget = dynamicBuffer;
        cachedPairs.resize(newWeights.size());
    } else {
        mode = QueryMode::Plain;
        remainingBudget = -1.0;
    }
}

void WeightedIndex::rebuildClasses() {
    const std::vector<double>& newWeights = *weights;
    const VecList& newPositions = *positions;

    invExpWeights.resize(newWeights.size());
#pragma omp parallel for default(none) shared(newWeights) schedule(static)
    for (size_t i = 0; i < newWeights.size(); i++) {
        invExpWeights[i] = 1.0 / Toolkit::myPow(newWeights[i], 1.0 / static_cast<double>(DIMENSION));
    }

    const double minWeight = *std::min_element(newWeights.begin(), newWeights.end());
    double maxWeight = *std::max_element(newWeights.begin(), newWeights.end());
    classBounds.clear();
    for (double bound = minWeight * doublingFactor; bound < maxWeight; bound *= doublingFactor) {
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
            case IndexType::Sprk:
                spacialIndices.push_back(std::make_unique<SprkQueries>(classContent[i], DIMENSION));
                break;
            default:
                LOG_ERROR("Unknown index type");
                break;
        }
    }
}

void WeightedIndex::getOwnedRepellingPairs(const NodeId v, std::vector<NodeId>& out) {
    ASSERT(positions != nullptr, "update() must run before queries");
    out.clear();
    const double weight = (*weights)[v];
    const double invV = invExpWeights[v];

    if (mode == QueryMode::Reuse) {
        std::vector<NodeId>& cache = cachedPairs[v];
        size_t keep = 0;
        for (const NodeId u : cache) {
            const double dist = vectorOperations::calculateLPNorm((*positions)[v], (*positions)[u]);
            const double weightedDist = dist * invV * invExpWeights[u];
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
    const double radiusSlack = mode == QueryMode::Fill ? dynamicBuffer : 0.0;
    for (size_t i = 0; i <= static_cast<size_t>(ownClass) && i < spacialIndices.size(); i++) {
        queryClass(i, (*positions)[v], weight, radiusSlack, candidates);
    }

    if (mode == QueryMode::Fill) {
        std::vector<NodeId>& cache = cachedPairs[v];
        cache.clear();
        for (const NodeId u : candidates) {
            if (u == v || !ownsPair(weight, (*weights)[u], v, u)) continue;
            const double dist = vectorOperations::calculateLPNorm((*positions)[v], (*positions)[u]);
            const double weightedDist = dist * invV * invExpWeights[u];
            // the class query radius over-covers light partners; keep only what can
            // reach the threshold within the buffer
            if (weightedDist >= 1.0 + dynamicBuffer * invV * invExpWeights[u]) continue;
            cache.push_back(u);
            if (weightedDist < 1.0) out.push_back(u);
        }
        return;
    }

    // mode == QueryMode::Plain
    for (const NodeId u : candidates) {
        if (u == v || !ownsPair(weight, (*weights)[u], v, u)) continue;
        // pairs at weighted distance >= 1 contribute zero force and zero loss
        const double dist = vectorOperations::calculateLPNorm((*positions)[v], (*positions)[u]);
        if (dist * invV * invExpWeights[u] >= 1.0) continue;
        out.push_back(u);
    }
}

void WeightedIndex::queryClass(const size_t weightClass, CVecRef p, const double weight, const double radiusSlack,
                               std::vector<NodeId>& output) const {
    ASSERT(spacialIndices.size() == maxWeightOfClass.size(), "Indices and weight classes must have the same size");
    ASSERT(weightClass < maxWeightOfClass.size());

    const double queryRadius =
        Toolkit::myPow(weight * maxWeightOfClass[weightClass], 1.0 / (double)DIMENSION) + radiusSlack;
    ASSERT(queryRadius > 0);

    thread_local std::vector<uint64_t> localIds;
    spacialIndices[weightClass]->query_sphere(p, queryRadius, localIds);
    for (const uint64_t localId : localIds) {
        output.push_back(classToGlobal[weightClass][localId]);
    }
}
