#include "EdgeDetection.hpp"

#include "EdgeSampler.hpp"
#include "Rand.hpp"

std::vector<std::string> EdgeDetection::getMetricValues() {
    const ll N = graph.getNumVertices();
    const ll M = graph.getNumEdges();
    const ll noM = (N * (N - 1) / 2) - M;

    std::vector<histEntry> histogram;
    int numSampledNonEdges;
    int numSampledEdges;

    // sample edges and non edges from the graph
    histInfo tmp = EdgeSampler::sampleHistEntries(graph, embedding, edgeSampleScale);
    histogram = tmp.histogramm;
    numSampledEdges = tmp.numEdges;
    numSampledNonEdges = tmp.numNonEdges;

    // find the optimal index that minimizes F1 score
    flt_t wrongEdgesPercent = 1.0;  // percent of how many edges are wrongly classified at the current index
    flt_t wrongNonEdgesPercent = 0.0;
    int bestF1Idx = -1;
    flt_t bestF1 = -1;
    flt_t bestPrecision = -1;
    flt_t bestRecall = -1;

    // find optimal position
    for (int i = 0; i < histogram.size(); i++) {
        if (histogram[i].isEdge) {  // is a edge
            wrongEdgesPercent -= 1.0 / numSampledEdges;
        } else {  // is not a neighbor
            wrongNonEdgesPercent += 1.0 / numSampledNonEdges;
        }

        // calculate current F1 score
        // see: https://en.wikipedia.org/wiki/F-score
        flt_t truePositives = (flt_t{1.0} - wrongEdgesPercent) * static_cast<flt_t>(M);
        flt_t retrievedElements = ((flt_t{1.0} - wrongEdgesPercent) * static_cast<flt_t>(M)) + (wrongNonEdgesPercent * static_cast<flt_t>(noM));
        flt_t relevantElemets = static_cast<flt_t>(M);

        flt_t precision = truePositives / retrievedElements;
        flt_t recall = truePositives / relevantElemets;
        flt_t F1 = flt_t{2.0} / (flt_t{1.0} / precision + flt_t{1.0} / recall);

        // std::vector<int> interestingKs{1, 2, 4, 8, 16, 32, 128, 265, 512};
        // if (std::find(interestingKs.begin(), interestingKs.end(), i) != interestingKs.end()) {
        //     std::cout << "Precision@" << i << " is " << precision << std::endl;
        // }

        if (F1 > bestF1) {
            bestF1Idx = i;
            bestF1 = F1;
            bestPrecision = precision;
            bestRecall = recall;
        }
    }

    LOG_DEBUG("Best best F1 at: " << bestF1Idx);
    unused(bestF1Idx);
    std::vector<std::string> result = {std::to_string(bestPrecision),  // precision
                                       std::to_string(bestRecall),     // recall
                                       std::to_string(bestF1)};        // F1-score
    return result;
}

std::vector<std::string> EdgeDetection::getMetricNames() {
    std::vector<std::string> result = {"precision",  // precision
                                       "recall",     // recall
                                       "edgeF1"};    // F1-score
    return result;
}
