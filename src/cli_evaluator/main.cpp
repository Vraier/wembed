#include <algorithm>
#include <vector>

#include "ConfigParser.hpp"
#include "EdgeDetection.hpp"
#include "EmbeddingIO.hpp"
#include "GeneralGraphInfo.hpp"
#include "Graph.hpp"
#include "GraphAlgorithms.hpp"
#include "GraphIO.hpp"
#include "Options.hpp"
#include "Reconstruction.hpp"
#include "TimeParser.hpp"

void addOptions(CLI::App& app, Options& opts);

int main(int argc, char* argv[]) {
    // Parse command line arguments
    CLI::App app("CLI Evaluator");
    Options options;
    addOptions(app, options);
    CLI11_PARSE(app, argc, argv);

    if (options.seed != -1) {
        Rand::setSeed(options.seed);
    }

    // read in graph
    Graph inputGraph = GraphIO::readEdgeList(options.edgeListPath, options.edgeListComment, options.edgeListDelimiter);
    if (!GraphAlgo::isConnected(inputGraph)) {
        LOG_ERROR("Graph is not connected");
        return 0;
    }

    // read in embedding
    std::vector<std::vector<double>> coords = EmbeddingIO::readCoordinatesFromFile(
        options.embeddingPath, options.embeddingComment, options.embeddingDelimiter);
    std::shared_ptr<Embedding> embedding = EmbeddingIO::parseEmbedding(options.embType, coords);
    if (embedding == nullptr) {
        LOG_ERROR("Embedding could not be parsed");
        return 0;
    }
    if (embedding->getDimension() == 0) {
        LOG_ERROR("Embedding dimension is 0");
        return 0;
    }

    // construct metrics
    std::vector<std::unique_ptr<Metric>> metrics;
    metrics.push_back(std::make_unique<GeneralGraphInfo>(inputGraph));
    metrics.push_back(std::make_unique<TimeParser>(options.timePath));
    metrics.push_back(std::make_unique<ConfigParser>(options.logPath, options.logType));
    metrics.push_back(std::make_unique<Reconstruction>(inputGraph, embedding, options.nodeSamplePercent));
    metrics.push_back(std::make_unique<EdgeDetection>(inputGraph, embedding, options.edgeSampleScale));

    // print the header for an svg file
    std::vector<std::string> valueNames;
    std::vector<std::string> tmpNames;

    valueNames.push_back("metric-type");

    for (auto& m : metrics) {
        tmpNames = m->getMetricNames();
        valueNames.insert(valueNames.end(), tmpNames.begin(), tmpNames.end());
    }
    Metric::printCSVToConsole(valueNames);


    // calculate and print the metrics for an svg file
    std::vector<std::string> valueMetrics;
    std::vector<std::string> tmpMetrics;

    valueMetrics.push_back(std::to_string(options.embType));

    for (auto& m : metrics) {
        tmpMetrics = m->getMetricValues();
        valueMetrics.insert(valueMetrics.end(), tmpMetrics.begin(), tmpMetrics.end());
    }

    Metric::printCSVToConsole(valueNames);
    Metric::printCSVToConsole(valueMetrics);
    
    return 0;
}

void addOptions(CLI::App& app, Options& options) {
    app.add_option("-g,--edge-list", options.edgeListPath, "Path to the edge list file")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--edge-list-comment", options.edgeListComment, "Comment symbol for the edge list file");
    app.add_option("--edge-list-delimiter", options.edgeListDelimiter, "Delimiter for the edge list file");
    app.add_option("-e,--embedding", options.embeddingPath, "Path to the embedding file")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--embedding-comment", options.embeddingComment, "Comment symbol for the embedding file");
    app.add_option("--embedding-delimiter", options.embeddingDelimiter, "Delimiter for the embedding file");
    app.add_option("-l,--log", options.logPath, "Path to the log file");
    app.add_option("-t,--time", options.timePath, "Path to the time file");

    app.add_option("--seed", options.seed, "Seed for the random number generator");
    app.add_option("--log-type", options.logType, "Type of log file");
    app.add_option("--edge-sample-factor", options.edgeSampleScale,
                   "Factor for how many more non edges get sampled than edges");
}