#include <CLI/CLI.hpp>

#include "EmbeddingIO.hpp"
#include "GirgGenerator.hpp"
#include "GraphIO.hpp"
#include "Macros.hpp"
#include "Options.hpp"
#include "Rand.hpp"

void addOptions(CLI::App& app, Options& opts);

int main(int argc, char* argv[]) {
    // Parse command line arguments
    CLI::App app("CLI Generator");
    Options options;
    addOptions(app, options);
    CLI11_PARSE(app, argc, argv);

    if (options.seed != -1) {
        Rand::setSeed(options.seed);
    }

    auto graphEmbedding = GirgGenerator::generateRandomGraph(options);

    // generate a girg/euclidean random graph
    GraphIO::writeToEdgeList(options.girgFile, std::get<0>(graphEmbedding));

    // write coordinates to file
    if (options.girgCoords != "") {
        EmbeddingIO::writeCoordinates(options.girgCoords, std::get<1>(graphEmbedding), std::get<2>(graphEmbedding));
    }

    return 0;
}

void addOptions(CLI::App& app, Options& options) {
    app.add_option("-o,--girg-file", options.girgFile, "Path to the output file for the edge list")->required();
    app.add_option("-w,--girg-coords", options.girgCoords,
                   "Path to the output file for the coordinates (including weights)");

    app.add_option("-s,--seed", options.seed, "Seed for the random number generator. -1 uses time as seed")
        ->capture_default_str();
    app.add_option("-n,--nodes", options.numNodes, "Maximum number of nodes")->capture_default_str();
    app.add_option("--ple", options.ple, "Power law exponent")->capture_default_str()->check(CLI::Range(2.0, std::numeric_limits<double>::infinity()));
    app.add_option("--avg-deg", options.averageDegree, "Average degree of the graph")->capture_default_str();
    app.add_option("-d,--gen-dim", options.genDimension, "Dimension of the generated graph")->capture_default_str()->check(CLI::Range(1, 5));
    app.add_option("-t,--temp", options.temperature, "Temperature for the girg")->capture_default_str()->check(CLI::Range(0.0, 1.0));
    app.add_flag("--torus", options.torus, "Generates the graph on the torus");
}
