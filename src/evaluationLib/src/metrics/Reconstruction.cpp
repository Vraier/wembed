#include "Reconstruction.hpp"
#include "Toolkit.hpp"

#include <fstream>

std::vector<std::string> Reconstruction::getMetricValues() {
    std::vector<double> constructAtDegVals;
    std::vector<double> averagePrecisions;

    std::vector<nodeEntry> hist = NodeSampler::sampleHistEntries(graph, embedding, numNodeSamples);
    for (auto e : hist) {
        constructAtDegVals.push_back(e.deg_precision);
        averagePrecisions.push_back(e.average_precision);
    }

    std::vector<std::string> result;
    result.push_back(std::to_string(Toolkit::averageFromVector(constructAtDegVals)));  // using deg as k
    result.push_back(std::to_string(Toolkit::averageFromVector(averagePrecisions)));   // mean average precision

    //writeHistogram(hist);
    
    return result;
}

std::vector<std::string> Reconstruction::getMetricNames() {
    std::vector<std::string> result;
    result.push_back("constructDeg");  // using deg as k
    result.push_back("MAP");           // mean average precision
    return result;
}

void Reconstruction::writeHistogram(const std::vector<nodeEntry>& entries) {
    std::ofstream output;
    output.open("node_histogram.csv");

    std::string firstLine = "NodeV,degV,degPrecision,avgPrecision";

    // header of csv file
    output << firstLine << std::endl;

    // output entries
    for (auto e : entries) {
        output << e.v << "," << e.degV << ","
               << e.deg_precision << "," << e.average_precision;
        output << std::endl;
    }
    output.close();
}
