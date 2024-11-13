#include "NodeSampler.hpp"

#include "Macros.hpp"

NodeHistogram NodeSampler::sampleHistEntries(const OptionValues& opts, const Graph& graph, Embedding& embedding,
                                             const std::vector<int>& kVals) {
    int N = graph.getNumVertices();
    std::vector<bool> isNeighbor(N, false);
    std::vector<int> nodePermutation = Rand::randomPermutation(N);
    const int numSampledNodes = std::min((int)(N * opts.nodeSamplePercent), std::min((int)N, 1000));
    NodeHistogram result;

    LOG_INFO( "Sampling " << numSampledNodes << " nodes");
    if (N - 1 < kVals.back()) {
        LOG_WARNING( "Not enough nodes in the graph to compute precision at " << kVals.back());
    }

    for (int i = 0; i < numSampledNodes; i++) {
        nodeEntry newEntry;

        const NodeId v = nodePermutation[i];
        const int degV = graph.getNeighbors(v).size();

        newEntry.v = v;
        newEntry.degV = degV;

        // remember which nodes are neighbor
        for (NodeId w : graph.getNeighbors(v)) {
            isNeighbor[w] = true;
        }

        // calculate the distance to every other node
        EdgeLengthToNode distances;
        for (NodeId x = 0; x < N; x++) {
            if (v == x) continue;
            distances.push_back(std::make_pair(embedding.getSimilarity(v, x), x));
        }

        // find the k nearest nodes
        std::sort(distances.begin(), distances.end());

        // determine construction value (at k)
        std::vector<double> precisions = getPrecisionsForNode(v, distances, isNeighbor);
        std::vector<double> recalls = getRecallsForNode(v, degV, distances, isNeighbor);
        newEntry.deg_precision = precisions[degV - 1];
        for (int i = 0; i < kVals.size(); i++) {
            const int k = kVals[i];
            if ((k - 1) > precisions.size()) {
                newEntry.k_to_precision[k] = precisions.back();
            } else {
                newEntry.k_to_precision[k] = precisions[k - 1];
            }
        }

        newEntry.average_precision = getAveragePrecision(v, distances, precisions, recalls, isNeighbor);
        // reset neighbors
        for (NodeId w : graph.getNeighbors(v)) {
            isNeighbor[w] = false;
        }

        result.push_back(newEntry);
    }

    LOG_INFO( "Finished sampling");

    return result;
}

std::vector<double> NodeSampler::getPrecisionsForNode(NodeId v, const EdgeLengthToNode& distances,
                                                      const std::vector<bool>& isNeighbor) {
    ASSERT(distances.size() + 1 == isNeighbor.size());
    unused(v);

    std::vector<double> precisions;

    int numCorrect = 0;
    int num_inserted = 0;
    for (int i = 0; i < distances.size(); i++) {
        if (isNeighbor[distances[i].second]) {
            numCorrect += 1;
        }
        num_inserted += 1;
        double precision = (double)numCorrect / (double)num_inserted;
        precisions.push_back(precision);
    }
    return precisions;
}

std::vector<double> NodeSampler::getRecallsForNode(NodeId v, int deg, const EdgeLengthToNode& distances,
                                                   const std::vector<bool>& isNeighbor) {
    ASSERT(distances.size() + 1 == isNeighbor.size());
    unused(v);

    std::vector<double> recalls;
    int numCorrect = 0;

    for (int i = 0; i < distances.size(); i++) {
        if (isNeighbor[distances[i].second]) {
            numCorrect += 1;
        }
        double recall = (double)numCorrect / (double)deg;
        recalls.push_back(recall);
    }
    return recalls;
}

double NodeSampler::getAveragePrecision(NodeId v, const EdgeLengthToNode& distances,
                                        const std::vector<double>& precisions, 
                                        const std::vector<double>& recalls,
                                        const std::vector<bool>& isNeighbor) {
    ASSERT(distances.size() == precisions.size());
    ASSERT(distances.size() + 1 == isNeighbor.size());
    unused(v);
    unused(recalls);

    std::vector<double> neighborPrecisions;
    //double lastRecall = 0;
    //double weightedRecallSum = 0;
    for (int i = 0; i < distances.size(); i++) {
        const NodeId u = distances[i].second;
        if (isNeighbor[u]) {
            neighborPrecisions.push_back(precisions[i]);

            //double recallDiff = recalls[i] - lastRecall;
            //weightedRecallSum += recallDiff * precisions[i];
            //lastRecall = recalls[i];
        }
    }

    return averageFromVector(neighborPrecisions);
    //return weightedRecallSum;
}

double NodeSampler::averageFromVector(const std::vector<double>& values) {
    double sum = 0;
    for (double v : values) {
        sum += v;
    }
    if (values.size() == 0) {
        return -1;
    } else {
        return sum / (double)values.size();
    }
}
