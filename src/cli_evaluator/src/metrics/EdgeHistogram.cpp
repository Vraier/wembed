#include "EdgeHistogram.hpp"

#include <fstream>

EdgeHistogram::EdgeHistogram(const OptionValues& opts, const Graph& g, Embedding& emb)
    : options(opts), graph(g), embedding(emb) {}

std::vector<std::string> EdgeHistogram::getMetricValues() {
    std::vector<histEntry> histogram;
    int numSampledNonEdges;
    int numSampledEdges;

    // sample edges and non edges from the graph
    histInfo tmp = EdgeSampler::sampleHistEntries(options, graph, embedding);
    histogram = tmp.hist;
    numSampledEdges = tmp.numEdges;
    numSampledNonEdges = tmp.numNonEdges;

    // find the optimal index that minimizes the average error or F1 score
    int bestAverageErrorIdx = -1;
    double wrongEdgesPercent = 1.0;  // percent of how many edges are wrongly classified at the current index
    double wrongNonEdgesPercent = 0.0;

    double bestEdgeError = -1;
    double bestNonEdgeError = -1;
    double bestAverageError = (wrongEdgesPercent + wrongNonEdgesPercent) / 2.0;

    // find optimal position
    for (int i = 0; i < histogram.size(); i++) {
        if (histogram[i].isEdge) {  // is a edge
            wrongEdgesPercent -= 1.0 / numSampledEdges;
        } else {  // is not a neighbor
            wrongNonEdgesPercent += 1.0 / numSampledNonEdges;
        }

        // calculate current average error
        double currAverageError = (wrongEdgesPercent + wrongNonEdgesPercent) / 2.0;

        if (currAverageError < bestAverageError) {
            bestAverageErrorIdx = i;

            bestAverageError = currAverageError;
            bestEdgeError = wrongEdgesPercent;
            bestNonEdgeError = wrongNonEdgesPercent;
        }
    }

    if (options.histFile != "") {
        writeHistogram(options, histogram);
    }

    LOG_INFO( "Best averageError at: " << bestAverageErrorIdx << std::endl);

    std::vector<std::string> result = {std::to_string(bestEdgeError),                                // edge error
                                       std::to_string(bestNonEdgeError),                             // non edge error
                                       std::to_string((bestEdgeError + bestNonEdgeError) / 2.0),     // average error
                                       std::to_string(std::abs(bestEdgeError - bestNonEdgeError))};  // balancing error

    return result;
}

std::vector<std::string> EdgeHistogram::getMetricNames() {
    std::vector<std::string> result = {
        "edge_error",
        "non_edge_error",
        "average_error",
        "balancing_error",
    };
    return result;
}

void EdgeHistogram::writeHistogram(const OptionValues& options, const std::vector<histEntry>& entries) {
    std::ofstream output;
    output.open(options.histFile);

    // header of csv file
    output << "NodeV,NodeW,wv,ww,distance,similarity,isEdge" << std::endl;

    // output entries
    for (histEntry e : entries) {
        output << e.v << "," << e.w << "," << e.weightV << "," << e.weightW << "," << e.distance << "," << e.similarity
               << "," << (e.isEdge ? "TRUE" : "FALSE") << std::endl;
    }
    output.close();
}
