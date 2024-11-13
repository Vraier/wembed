#include <algorithm>
#include <vector>

#include "ConfigParser.hpp"
#include "EdgeDetection.hpp"
#include "GeneralGraphInfo.hpp"
#include "EdgeHistogram.hpp"
#include "EmbeddingIO.hpp"
#include "EvalOptions.hpp"
#include "Graph.hpp"
#include "GraphAlgorithms.hpp"
#include "GraphIO.hpp"
#include "Reconstruction.hpp"
#include "TimeParser.hpp"

int main(int argc, char* argv[]) {
    EvalOptions options;
    OptionValues optionValues;
    options.parser.parseCommandLine(argc, argv);
    optionValues = options.values;

    if (!optionValues.printMetricNames) {
        options.parser.printVariableMap();
    }

    if (optionValues.help) {
        options.parser.printHelp();
        return 0;
    }

    if (optionValues.seed != -1) {
        Rand::setSeed(optionValues.seed);
    }

    // read in graph (if there is any)
    Graph tmpG;
    idMapping mapping;
    if (optionValues.edgeListFile != "") {
        std::tie(tmpG, mapping) = GraphIO::readEdgeList(optionValues.edgeListFile);
    } else if (!optionValues.printMetricNames) {
        LOG_ERROR( "No graph file provided. Terminating...");
        return 0;
    }

    // check if graph is connected
    Graph g;
    if (!optionValues.printMetricNames) {
        g = GraphAlgo::getLargestComponent(tmpG);
        if (tmpG.getNumVertices() != g.getNumVertices()) {
            LOG_ERROR( "Graph contains more than one component. Terminating...");
            return 0;
        }
    }

    // read in embedding (if there is any)
    Embedding* emb = nullptr;
    if (optionValues.coordFile != "") {
        auto coords = EmbeddingIO::readCoordinatesFromFile(optionValues.coordFile, mapping, optionValues.coordComment,
                                                           optionValues.coordDelimiter);
        emb = EmbeddingIO::parseEmbedding(static_cast<EmbeddingType>(optionValues.embType), coords);
        optionValues.embDimension = emb->getDimension();
    } else if (!optionValues.printMetricNames) {
        LOG_ERROR( "No embedding file provided. Terminating...");
        return 0;
    }
    if (emb != nullptr && emb->getDimension() == 0) {
        LOG_ERROR( "Embedding dimension is 0");
        return 0;
    }

    // construct metrics
    std::vector<Metric*> metrics;
    metrics.push_back(new GeneralGraphInfo(optionValues, g));
    metrics.push_back(new TimeParser(optionValues));
    metrics.push_back(new ConfigParser(optionValues));
    metrics.push_back(new EdgeHistogram(optionValues, g, *emb));
    metrics.push_back(new Reconstruction(optionValues, g, *emb));
    metrics.push_back(new EdgeDetection(optionValues, g, *emb));

    // print the header for an svg file
    std::vector<std::string> valueNames;
    std::vector<std::string> tmpNames;

    valueNames.push_back("metric-type");

    for (auto m : metrics) {
        tmpNames = m->getMetricNames();
        valueNames.insert(valueNames.end(), tmpNames.begin(), tmpNames.end());
    }
    Metric::printCSVToConsole(valueNames);

    // if we should only print the header we can stop here
    if (optionValues.printMetricNames) {
        for (auto m : metrics) {
            delete m;
        }
        delete emb;
        return 0;
    }

    // calculate and print the metrics for an svg file
    if (optionValues.edgeListFile != "" && optionValues.coordFile != "") {
        std::vector<std::string> valueMetrics;
        std::vector<std::string> tmpMetrics;

        valueMetrics.push_back(std::to_string(optionValues.embType));

        for (auto m : metrics) {
            tmpMetrics = m->getMetricValues();
            valueMetrics.insert(valueMetrics.end(), tmpMetrics.begin(), tmpMetrics.end());
        }

        Metric::printCSVToConsole(valueNames);
        Metric::printCSVToConsole(valueMetrics);
    }

    for (auto m : metrics) {
        delete m;
    }
    delete emb;
    return 0;
}
