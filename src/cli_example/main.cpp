#include <iostream>

#include "EmbeddingIO.hpp"
#include "GraphAlgorithms.hpp"
#include "GraphIO.hpp"
#include "Options.hpp"
#include "SimpleSamplingEmbedder.hpp"

void addOptions(CLI::App& app, Options& opts);

int main(int argc, char* argv[]) {
    // Parse the command line arguments
    CLI::App app("Embedder CLI");
    Options opts;
    addOptions(app, opts);
    CLI11_PARSE(app, argc, argv);

    // Read the graph
    Graph inputGraph = GraphIO::readEdgeList(opts.graphPath);
    if (!GraphAlgo::isConnected(inputGraph)) {
        LOG_ERROR("Graph is not connected");
        return 0;
    }

    // Embed the graph
    SimpleSamplingEmbedder embedder(inputGraph, opts.embedderOptions);

    if (opts.draw) {
        while (!embedder.isFinished()) {
            embedder.calculateStep();
        }
    } else {
        embedder.calculateEmbedding();
    }

    // Output timings
    if (opts.showTimings) {
        LOG_INFO("Printing Timings");
        std::vector<util::TimingResult> timings = embedder.getTimings();
        std::cout << util::timingsToStringRepresentation(timings);
    }

    // Output the embedding
    if (opts.embeddingPath != "") {
        std::vector<std::vector<double>> coordinates = embedder.getCoordinates();
        std::vector<double> weights = embedder.getWeights();
        EmbeddingIO::writeCoordinates(opts.embeddingPath, coordinates, weights);
    }

    return 0;
}

void addOptions(CLI::App& app, Options& opts) {
    app.add_option("-i,--graph", opts.graphPath, "Path to the graph file")->required()->check(CLI::ExistingFile);
    app.add_option("-o,--embedding", opts.embeddingPath, "Path to the output embedding file");
    app.add_flag("-d,--draw", opts.draw, "Animates the graph embedding. Slow but helpful for debugging.");
    app.add_flag("--timings", opts.showTimings, "Print timings after embedding");

    // Embedder Options
    app.add_option("--dim-hint", opts.embedderOptions.dimensionHint, "Dimension hint")->capture_default_str();
    app.add_option("--dim", opts.embedderOptions.embeddingDimension, "Embedding dimension")->capture_default_str();
}