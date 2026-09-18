#include "WeightedIndex.hpp"

#include "SprkQueries.hpp"
#include "VectorOperations.hpp"

void WeightedIndex::update(const VecList& newPositions, const std::vector<double>& newWeights) {
    ASSERT(newPositions.size() == newWeights.size(), "Positions and weights must have the same size");
    ASSERT(newPositions.dimension() == DIMENSION, "Positions must have the same dimension as the index");
    this->positions = &newPositions;
    this->weights = &newWeights;

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

void WeightedIndex::getOwnedRepellingPairs(const NodeId v, std::vector<NodeId>& out) const {
    ASSERT(positions != nullptr, "update() must run before queries");
    thread_local std::vector<NodeId> candidates;
    candidates.clear();

    const double weight = (*weights)[v];
    const auto ownClass = std::upper_bound(classBounds.begin(), classBounds.end(), weight) - classBounds.begin();
    for (size_t i = 0; i <= static_cast<size_t>(ownClass) && i < spacialIndices.size(); i++) {
        queryClass(i, (*positions)[v], weight, candidates);
    }

    out.clear();
    for (const NodeId u : candidates) {
        if (u == v || !ownsPair(weight, (*weights)[u], v, u)) continue;
        // pairs at weighted distance >= 1 contribute zero force and zero loss
        const double dist = vectorOperations::calculateLPNorm((*positions)[v], (*positions)[u]);
        if (dist * invExpWeights[v] * invExpWeights[u] >= 1.0) continue;
        out.push_back(u);
    }
}

void WeightedIndex::queryClass(const size_t weightClass, CVecRef p, const double weight,
                               std::vector<NodeId>& output) const {
    ASSERT(spacialIndices.size() == maxWeightOfClass.size(), "Indices and weight classes must have the same size");
    ASSERT(weightClass < maxWeightOfClass.size());

    const double queryRadius = Toolkit::myPow(weight * maxWeightOfClass[weightClass], 1.0 / (double)DIMENSION);
    ASSERT(queryRadius > 0);

    thread_local std::vector<uint64_t> localIds;
    spacialIndices[weightClass]->query_sphere(p, queryRadius, localIds);
    for (const uint64_t localId : localIds) {
        output.push_back(classToGlobal[weightClass][localId]);
    }
}
