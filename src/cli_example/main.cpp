#include <iostream>
#include <omp.h>

#include "EmbeddingIO.hpp"
#include "GraphAlgorithms.hpp"
#include "GraphIO.hpp"
#include "Options.hpp"
#include "SimpleSamplingEmbedder.hpp"

#ifdef EMBEDDING_USE_ANIMATION
#include "SFMLDrawer.hpp"
#endif
#include "SVGDrawer.hpp"

void addOptions(CLI::App& app, Options& opts);

int main(int argc, char* argv[]) {
    std::cout << "Using OpenMP with " << omp_get_max_threads() << " threads" << std::endl;

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

    #ifdef EMBEDDING_USE_ANIMATION
    if (opts.animate) {
        SFMLDrawer drawer;
        while(!embedder.isFinished()) {
            embedder.calculateStep();
            drawer.processFrame(inputGraph, embedder.getCoordinates());
        }
    } else {
        embedder.calculateEmbedding();
    }
    #else
    embedder.calculateEmbedding();
    #endif

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

    // Output the SVG
    if (opts.svgPath != "") {
        std::vector<std::vector<double>> coordinates = embedder.getCoordinates();
        std::vector<double> weights = embedder.getWeights();
        SVGOutputWriter svgWriter;
        svgWriter.write(opts.svgPath, inputGraph, coordinates);
    }

    return 0;
}

void addOptions(CLI::App& app, Options& opts) {
    // Input / Output
    app.add_option("-i,--graph", opts.graphPath, "Path to the graph file")->required()->check(CLI::ExistingFile);
    app.add_option("-o,--embedding", opts.embeddingPath, "Path to the output embedding file");
    app.add_flag("--timings", opts.showTimings, "Print timings after embedding");

    // Visualization
    app.add_option("--svg", opts.svgPath, "Path to the output svg file");
    #ifdef EMBEDDING_USE_ANIMATION
    app.add_flag("--animate", opts.animate, "Animate the embedding, only avaliable if compiled with SFML");
    #endif

    // Embedder Options
    app.add_option("--dim-hint", opts.embedderOptions.dimensionHint, "Dimension hint")->capture_default_str();
    app.add_option("--dim", opts.embedderOptions.embeddingDimension, "Embedding dimension")->capture_default_str();
    app.add_option("--iterations", opts.embedderOptions.maxIterations, "Maximum number of iterations")->capture_default_str();
    app.add_option("--cooling", opts.embedderOptions.coolingFactor, "Cooling during gradient descent")->capture_default_str();
    app.add_option("--speed", opts.embedderOptions.speed, "Speed of the embedding process")->capture_default_str();
}