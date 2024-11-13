#include "DrawOptions.hpp"
#include "EmbeddingIO.hpp"
#include "GraphIO.hpp"
#include "IPEDrawer.hpp"
#include "SVGDrawer.hpp"
#include "VecList.hpp"
#include "Macros.hpp"

int main(int argc, char *argv[]) {
    DrawOptions options;
    options.parser.parseCommandLine(argc, argv);
    options.parser.printVariableMap();
    OptionValues optionValues = options.values;

    if (optionValues.help) {
        options.parser.printHelp();
        return 0;
    }

    Graph g;
    if (optionValues.edgPath != "") {
        g = GraphIO::readEdgeList(optionValues.edgPath).first;
    } else {
        LOG_WARNING( "No graph file provided. Terminating...");
        return 0;
    }

    std::vector<std::vector<double>> coordinates;
    std::vector<double> weights;
    if (optionValues.coordFile != "") {
        coordinates = EmbeddingIO::readCoordinatesFromFile(optionValues.coordFile);
    } else if (optionValues.weightCoordFile != "") {
        auto tmp = EmbeddingIO::readCoordinatesFromFile(optionValues.weightCoordFile);
        auto res = EmbeddingIO::splitLastColumn(tmp);
        coordinates = res.first;
        weights = res.second;
    } else {
        LOG_WARNING( "No coordinate file provided. Terminating...");
        return 0;
    }
    ASSERT(coordinates.size() == g.getNumVertices(), "Number of vertices in graph and coordinates file do not match.");

    VecList vecCoordinates(coordinates[0].size());
    vecCoordinates.setSize(coordinates.size());
    for (int i = 0; i < coordinates.size(); i++) {
        for (int d = 0; d < coordinates[i].size(); d++) {
            vecCoordinates[i][d] = coordinates[i][d];
        }
    }

    if (optionValues.svgPath != "") {
        SVGOutputWriter writer;
        if (weights.size() == 0) {
            writer.write(optionValues.svgPath, g, vecCoordinates.convertToVector());
        } else {
            writer.write(optionValues.svgPath, g, vecCoordinates.convertToVector(), weights);
        }
    } if (optionValues.ipePath != "") {
        IpeOutputWriter writer(optionValues.ipePath, 500, 50);
        writer.write_graph(g, vecCoordinates.convertToVector());
    } if (optionValues.svgPath == "" && optionValues.ipePath == "") {
        LOG_WARNING( "No output path provided. Terminating...");
        return 0;
    }

    return 0;
}
