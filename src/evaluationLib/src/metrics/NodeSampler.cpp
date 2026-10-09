#include "NodeSampler.hpp"

#include "Macros.hpp"

std::vector<nodeEntry> NodeSampler::sampleHistEntries(const Graph& graph, std::shared_ptr<Embedding> embedding,
                                                      int numNodeSamples) {
    int N = graph.getNumVertices();
    numNodeSamples = std::min(numNodeSamples, N);                   // sample at most N nodes
    std::vector<int> nodePermutation = Rand::randomPermutation(N);  // used to sample random nodes
    std::vector<bool> isNeighbor(N, false);                         // reused for every node

    std::vector<nodeEntry> result(numNodeSamples);

#pragma omp parallel for firstprivate(isNeighbor), schedule(runtime)
    for (int i = 0; i < numNodeSamples; i++) {
        nodeEntry newEntry;

        const NodeId v = nodePermutation[i];
        const int degV = graph.getNeighbors(v).size();

        newEntry.v = v;
        newEntry.degV = degV;

        // remember which nodes are neighbors
        for (NodeId w : graph.getNeighbors(v)) {
            isNeighbor[w] = true;
        }

        // calculate the distance to every other node
        EdgeLengthToNode distances;
        for (NodeId x = 0; x < N; x++) {
            if (v == x) continue;
            distances.push_back(std::make_pair(embedding->getSimilarity(v, x), x));
        }

        // find the k nearest nodes
        std::sort(distances.begin(), distances.end());

        // determine construction value (at k)
        std::vector<flt_t> precisions = getPrecisionsForNode(v, distances, isNeighbor);
        std::vector<flt_t> recalls = getRecallsForNode(v, degV, distances, isNeighbor);
        newEntry.deg_precision = precisions[degV - 1];
        newEntry.average_precision = getAveragePrecision(v, distances, precisions, recalls, isNeighbor);
        // reset neighbors
        for (NodeId w : graph.getNeighbors(v)) {
            isNeighbor[w] = false;
        }
        result[i] = newEntry;
    }

    return result;
}

std::vector<flt_t> NodeSampler::getPrecisionsForNode(NodeId v, const EdgeLengthToNode& distances,
                                                      const std::vector<bool>& isNeighbor) {
    ASSERT(distances.size() + 1 == isNeighbor.size());
    unused(v);

    std::vector<flt_t> precisions;

    int numCorrect = 0;
    int num_inserted = 0;
    for (int i = 0; i < distances.size(); i++) {
        if (isNeighbor[distances[i].second]) {
            numCorrect += 1;
        }
        num_inserted += 1;
        flt_t precision = (flt_t)numCorrect / (flt_t)num_inserted;
        precisions.push_back(precision);
    }
    return precisions;
}

std::vector<flt_t> NodeSampler::getRecallsForNode(NodeId v, int deg, const EdgeLengthToNode& distances,
                                                   const std::vector<bool>& isNeighbor) {
    ASSERT(distances.size() + 1 == isNeighbor.size());
    unused(v);

    std::vector<flt_t> recalls;
    int numCorrect = 0;

    for (int i = 0; i < distances.size(); i++) {
        if (isNeighbor[distances[i].second]) {
            numCorrect += 1;
        }
        flt_t recall = static_cast<flt_t>(numCorrect) / static_cast<flt_t>(deg);
        recalls.push_back(recall);
    }
    return recalls;
}

flt_t NodeSampler::getAveragePrecision(NodeId v, const EdgeLengthToNode& distances,
                                        const std::vector<flt_t>& precisions, const std::vector<flt_t>& recalls,
                                        const std::vector<bool>& isNeighbor) {
    ASSERT(distances.size() == precisions.size());
    ASSERT(distances.size() + 1 == isNeighbor.size());
    unused(v);
    unused(recalls);

    std::vector<flt_t> neighborPrecisions;
    for (int i = 0; i < distances.size(); i++) {
        const NodeId u = distances[i].second;
        if (isNeighbor[u]) {
            neighborPrecisions.push_back(precisions[i]);
        }
    }
    return Toolkit::averageFromVector(neighborPrecisions);
}
