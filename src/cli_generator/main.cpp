#include "GenOptions.hpp"
#include "GirgGenerator.hpp"
#include "Graph.hpp"
#include "GraphAlgorithms.hpp"
#include "GraphIO.hpp"
#include "Macros.hpp"
#include "Rand.hpp"

void printCSVToConsole(const std::vector<std::string>& content);

int main(int argc, char* argv[]) {
    GenOptions options;
    options.parser.parseCommandLine(argc, argv);

    if (!options.checkValidity()) {
        options.parser.printHelp();
        return 0;
    }

    options.parser.printVariableMap();

    if (options.values.seed != -1) {
        Rand::setSeed(options.values.seed);
    }

    // generate a girg/euclidean random graph
    if (options.values.girgFile != "") {
        Graph g = GirgGenerator::generateRandomGraph(options.values);
        g = GraphAlgo::getLargestComponent(g);
        GraphIO::writeToEdgeList(options.values.girgFile, g);
    }

    return 0;
}
