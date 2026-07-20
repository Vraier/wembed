#include <iostream>

#include "Options.hpp"
#include "wembed.h"

void addOptions(CLI::App& app, Options& opts);

int main(int argc, char* argv[]) {
    CLI::App app("Embedder CLI");
    Options opts;
    addOptions(app, opts);
    CLI11_PARSE(app, argc, argv);

    if (opts.seed != -1) {
        wembed::setSeed(opts.seed);
    }

    wembed::Graph graph = wembed::graphFromEdgeListFile(opts.graphPath);

    wembed::Embedder embedder = wembed::createEmbedder(graph, opts.embedderOptions);

    if (!opts.inputEmbeddingPath.empty()) {
        auto coords = wembed::readCoordinatesFromFile(
            opts.inputEmbeddingPath, opts.embeddingComment, opts.embeddingDelimiter);
        embedder.setCoordinates(coords);
    }

    embedder.calculateEmbedding();

    if (opts.showTimings) {
        std::cout << wembed::timingsToString(embedder.getTimings());
    }

    if (!opts.embeddingPath.empty()) {
        embedder.writeCoordinates(opts.embeddingPath);
    }
    return 0;
}

void addOptions(CLI::App& app, Options& opts) {
    // Input / Output
    app.add_option("-i,--graph", opts.graphPath, "Path to an edge list")
        ->required()->check(CLI::ExistingFile);
    app.add_option("-o,--embedding", opts.embeddingPath, "Path to the output embedding file");
    app.add_option("--init-coordinates", opts.inputEmbeddingPath,
                   "Path to a file containing initial coordinates. If empty, coordinates are initialized randomly.");
    app.add_flag("--timings", opts.showTimings, "Print timings after embedding");

    // Embedder Options
    app.add_option("--seed", opts.seed, "Seed used during embedding. '-1' uses time as seed")
        ->capture_default_str();
    app.add_flag("--layered", opts.embedderOptions.layeredEmbedding, "Use layered embedding");
    app.add_option("--dim", opts.embedderOptions.embeddingDimension, "Embedding dimension")
        ->capture_default_str();
    app.add_option("--dim-hint", opts.embedderOptions.dimensionHint,
                   "Dimension hint. Negative values use dim as dimension hint.")
        ->capture_default_str();
    app.add_flag("--unit-weights", opts.embedderOptions.useUnitWeights,
                 "Disable degree-based weights (use unit weights instead)");
    app.add_option("--index-type", opts.embedderOptions.indexType,
                   "Type of spatial index used for the embedding (1=SNN, 2=Sprk)")
        ->capture_default_str();
    app.add_option("--min-change", opts.embedderOptions.positionMinChange,
                   "Minimum change in position to stop the embedding.")
        ->capture_default_str();
    app.add_option("--attraction", opts.embedderOptions.attractionScale,
                   "Changes magnitude of attracting forces")
        ->capture_default_str();
    app.add_option("--repulsion", opts.embedderOptions.repulsionScale,
                   "Changes magnitude of repulsing forces")
        ->capture_default_str();
    app.add_option("--centre,--center", opts.embedderOptions.centreScale,
                   "Strength of the centre-pull force. Useful for unconnected graphs (try ~0.01– 0.1). Default 0 disables it.")
        ->capture_default_str();
    app.add_option("--expansion", opts.embedderOptions.expansionStretch,
                   "Determines how much the embedding is stretched during layer expansion.")
        ->capture_default_str();
    app.add_option("--iterations", opts.embedderOptions.maxIterations, "Maximum number of iterations")
        ->capture_default_str();
    app.add_option("--cooling", opts.embedderOptions.coolingFactor, "Cooling during gradient descent")
        ->capture_default_str();
    app.add_option("--speed", opts.embedderOptions.learningRate, "Learning rate of the embedding process")
        ->capture_default_str();
}
