#include <omp.h>

#include <iostream>

#include "EmbeddingIO.hpp"
#include "GraphAlgorithms.hpp"
#include "GraphIO.hpp"
#include "LabelPropagation.hpp"
#include "LayeredEmbedder.hpp"
#include "Options.hpp"
#include "StringManipulation.hpp"
#include "WEmbedEmbedder.hpp"

#ifdef EMBEDDING_USE_ANIMATION
#include "SFMLDrawer.hpp"
#endif
#include "SVGDrawer.hpp"

void addOptions(CLI::App& app, Options& opts);

int main(int argc, char* argv[]) {
    // Parse the command line arguments
    CLI::App app("Embedder CLI");
    Options opts;
    addOptions(app, opts);
    CLI11_PARSE(app, argc, argv);

    // set the seed
    if (opts.seed != -1) {
        Rand::setSeed(opts.seed);
    }

    // Read the graph
    Graph inputGraph;
    if (opts.bipartite) {
        inputGraph = GraphIO::readBipartiteEdgeList(opts.graphPath);
    } else {
        inputGraph = GraphIO::readEdgeList(opts.graphPath);
    }
    if (!GraphAlgo::isConnected(inputGraph)) {
        LOG_ERROR("Graph is not connected");
        return 0;
    }

    // Embed the graph
    std::unique_ptr<EmbedderInterface> embedder;
    if (opts.layeredEmbedding) {
        LabelPropagation coarsener(PartitionerOptions{}, inputGraph,
                                   std::vector<double>(inputGraph.getNumEdges() * 2, 1.0));
        embedder = std::make_unique<LayeredEmbedder>(inputGraph, coarsener, opts.embedderOptions);
    } else {
        embedder = std::make_unique<WEmbedEmbedder>(inputGraph, opts.embedderOptions);
    }

    // SimpleSamplingEmbedder embedder(inputGraph, opts.embedderOptions);

#ifdef EMBEDDING_USE_ANIMATION
    if (opts.animate) {
        SFMLDrawer drawer;
        while (!embedder->isFinished()) {
            embedder->calculateStep();
            Graph currentGraph = embedder->getCurrentGraph();
            std::vector<std::vector<double>> coordinates = embedder->getCoordinates();
            std::vector<std::vector<double>> projection = Common::projectOntoPlane(coordinates);
            drawer.processFrame(currentGraph, projection);
        }
    } else {
        embedder->calculateEmbedding();
    }
#else
    embedder->calculateEmbedding();
#endif

    // Output timings
    if (opts.showTimings) {
        LOG_INFO("Printing Timings");
        std::vector<util::TimingResult> timings = embedder->getTimings();
        std::cout << util::timingsToStringRepresentation(timings);
    }

    // Output the embedding
    if (opts.embeddingPath != "") {
        std::vector<std::vector<double>> coordinates = embedder->getCoordinates();
        std::vector<double> weights = embedder->getWeights();
        EmbeddingIO::writeCoordinates(opts.embeddingPath, coordinates, weights);
    }
    return 0;
}

void addOptions(CLI::App& app, Options& opts) {
    // Input / Output
    app.add_option("-i,--graph", opts.graphPath, "Path to the graph file")->required()->check(CLI::ExistingFile);
    app.add_flag("--bipartite", opts.bipartite, "Treat the input graph as bipartite");
    app.add_option("-o,--embedding", opts.embeddingPath, "Path to the output embedding file");
    app.add_flag("--timings", opts.showTimings, "Print timings after embedding");

    // Visualization
#ifdef EMBEDDING_USE_ANIMATION
    app.add_flag("--animate", opts.animate, "Animate the embedding, only avaliable if compiled with SFML");
#endif

    // Embedder Options
    app.add_option("--seed", opts.seed, "Seed used during embedding. '-1' uses time as seed")->capture_default_str();
    app.add_flag("--layered", opts.layeredEmbedding, "Use layered embedding");
    app.add_option("--dim-hint", opts.embedderOptions.dimensionHint, "Dimension hint")->capture_default_str();
    app.add_option("--dim", opts.embedderOptions.embeddingDimension, "Embedding dimension")->capture_default_str();
    app.add_option("--weight-type", opts.embedderOptions.weightType,
                   "Affects the initial weights: " + util::mapToString(weightTypeMap))
        ->capture_default_str();
    app.add_option("--iterations", opts.embedderOptions.maxIterations, "Maximum number of iterations")
        ->capture_default_str();
    app.add_option("--cooling", opts.embedderOptions.coolingFactor, "Cooling during gradient descent")
        ->capture_default_str();
    app.add_option("--speed", opts.embedderOptions.speed, "Speed of the embedding process")->capture_default_str();
    app.add_flag("--use-inf-norm", opts.embedderOptions.useInfNorm,
                 "Uses L_inf norm instead of L_2 to calculate distance between vertices.");
}