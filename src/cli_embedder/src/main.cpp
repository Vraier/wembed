#include "DrawCommon.hpp"
#include "EmbOptions.hpp"
#include "EmbedderFactory.hpp"
#include "EmbeddingIO.hpp"
#include "GeometricGraphSampler.hpp"
#include "GraphAlgorithms.hpp"
#include "GraphIO.hpp"
#include "HierarchyEmbedder.hpp"
#include "Macros.hpp"
#include "Rand.hpp"
#include "SVGDrawer.hpp"
#include "Timings.hpp"
#include "Toolkit.hpp"
#include "VecList.hpp"

int main(int argc, char* argv[]) {
    EmbOptions options;
    OptionValues optionValues;
    options.parser.parseCommandLine(argc, argv);
    options.fillDefaultValues();
    options.parser.printVariableMap();
    optionValues = options.values;

    if (!options.checkValidity()) {
        options.parser.printHelp();
        return 0;
    }

    if (optionValues.seed != -1) {
        Rand::setSeed(optionValues.seed);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // read/generate the graph that will be used during embedding
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    Graph inputGraph;
    IdMapping mapping;
    if (optionValues.generatorOptions.graphPath != "") {
        std::pair<Graph, IdMapping> graphAndMapping =
            GraphIO::readGraph(static_cast<GraphFileType>(optionValues.generatorOptions.graphFileType),
                               optionValues.generatorOptions.graphPath);
        inputGraph = graphAndMapping.first;
        mapping = graphAndMapping.second;
    } else {
        // Generate a random 2D graph. this code is slow and is only for testing
        GeometricGraphSampler sampler;
        auto graphCoords = sampler.generateRandomGraphWithCoordinates(optionValues.generatorOptions.graphSize);
        inputGraph = graphCoords.first;
        mapping = Toolkit::createIdentity(graphCoords.first.getNumVertices());
    }

    if (!GraphAlgo::isConnected(inputGraph)) {
        LOG_ERROR( "Graph is not connected");
        return 0;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // check if we have girg weights available
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<double> initialWeights;
    std::vector<std::vector<double>> weightedCoordinates;
    if (optionValues.generatorOptions.girgWeights != "") {
        weightedCoordinates = EmbeddingIO::readCoordinatesFromFile(optionValues.generatorOptions.girgWeights);
        for (auto x : weightedCoordinates) {
            initialWeights.push_back(x.back());
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Embed the graph
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    VecList resultPositions(optionValues.dimension);
    std::vector<double> resultWeights;
    EmbeddedGraph embedGraph(optionValues.dimension, Graph());
    std::vector<util::TimingResult> timings;

    if (optionValues.embedderOptions.embedderType >= 5) {
        // Single level approach
        std::unique_ptr<AbstractSimpleEmbedder> slowEmbedder(
            EmbedderFactory::constructSimpleEmbedder(optionValues, inputGraph));
        auto cAndW = EmbeddingIO::splitLastColumn(weightedCoordinates);
        slowEmbedder->setWeightedCoordinates(cAndW.first, cAndW.second);
        slowEmbedder->initializeNewRun();
        slowEmbedder->calculateLayout();
        embedGraph = slowEmbedder->getEmbeddedGraph();
        timings = slowEmbedder->getTimings();
    } else {
        // Hierarchical approach
        HierarchyEmbedder embedder(optionValues, inputGraph);
        embedder.setInitialWeights(initialWeights);  // only relevant if original weights should be used
        embedder.initializeNewRun();
        embedder.calculateLayout();
        embedGraph = embedder.getEmbeddedGraph();
    }

    resultPositions = embedGraph.coordinates;
    resultWeights = embedGraph.getAllNodeWeights();

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // print timings
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    if (optionValues.show_timings) {
        std::cout << "Embedding Timings" << std::endl;
        std::cout << util::timingsToStringRepresentation(timings);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // output the desired information to file
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    if (optionValues.writerOptions.svgPath != "") {
        SVGOutputWriter writer;
        writer.write(optionValues.writerOptions.svgPath, inputGraph,
                     Common::projectOntoPlane(resultPositions.convertToVector()));
    }
    if (optionValues.writerOptions.coordPath != "") {
        if (optionValues.embedderOptions.embedderType == 11 || optionValues.embedderOptions.embedderType == 14) {
            // for the F2V embedder we use a dotProduct embedding
            EmbeddingIO::writeCoordinates(optionValues.writerOptions.coordPath, resultPositions, mapping);
        } else {
            EmbeddingIO::writeCoordinates(optionValues.writerOptions.coordPath, resultPositions, resultWeights, mapping);
        }
    }
    return 0;
}
