#include "Reconstruction.hpp"

#include <fstream>

Reconstruction::Reconstruction(const OptionValues& opts, const Graph& g, Embedding& emb) : options(opts),
                                                                                           graph(g),
                                                                                           embedding(emb),
                                                                                           buffer(options.embDimension) {}

std::vector<std::string> Reconstruction::getMetricValues() {
    std::vector<double> constructAtDegVals;
    std::vector<double> averagePrecisions;

    NodeHistogram hist = NodeSampler::sampleHistEntries(options, graph, embedding, kVals);
    for (auto e : hist) {
        constructAtDegVals.push_back(e.deg_precision);
        averagePrecisions.push_back(e.average_precision);
    }

    if (options.nodeHistFile != "") {
        writeHistogram(options, hist);
    }

    std::vector<std::string> result;
    result.push_back(std::to_string(NodeSampler::averageFromVector(constructAtDegVals)));  // using deg as k
    result.push_back(std::to_string(NodeSampler::averageFromVector(averagePrecisions)));   // mean average precision

    std::cout << "Neighbors of 0 ";
    for (int n : graph.getNeighbors(0)) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    return result;
}

std::vector<std::string> Reconstruction::getMetricNames() {
    std::vector<std::string> result;
    result.push_back("constructDeg");  // using deg as k
    result.push_back("MAP");           // mean average precision
    return result;
}

void Reconstruction::writeHistogram(const OptionValues& options, const NodeHistogram& entries) {
    std::ofstream output;
    output.open(options.nodeHistFile);

    std::string firstLine = "NodeV,degV,degPrecision,avgPrecision";
    for (int k : kVals) {
        firstLine += ",precisionAt" + std::to_string(k);
    }

    // header of csv file
    output << firstLine << std::endl;

    // output entries
    for (auto e : entries) {
        output << e.v << "," << e.degV << ","
               << e.deg_precision << "," << e.average_precision;
        for (int k : kVals) {
            output << "," << e.k_to_precision[k];
        }
        output << std::endl;
    }
    output.close();
}
