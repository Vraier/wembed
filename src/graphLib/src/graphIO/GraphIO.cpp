#include "GraphIO.hpp"

#include <fstream>
#include <stdexcept>

#include "StringManipulation.hpp"

Graph GraphIO::readEdgeList(std::string filePath, std::string comment, std::string delimiter) {
    std::ifstream input(filePath);
    if (!input.good()) {
        throw std::runtime_error("could not open edge list " + filePath);
    }

    std::vector<std::pair<NodeId, NodeId>> graphEdges;
    std::string line;
    int lineNumber = 0;
    while (std::getline(input, line)) {
        lineNumber++;
        if (util::startsWith(line, comment) || line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }
        const auto invalidLine = [&](const std::string& reason) {
            return std::invalid_argument(filePath + ":" + std::to_string(lineNumber) + ": " + reason + ": '" +
                                         line + "'");
        };

        // splitIntoTokens consumes its argument, keep line intact for the error message
        std::string rest = line;
        std::vector<std::string> tokens = util::splitIntoTokens(rest, delimiter);
        if (tokens.size() < 2) {
            throw invalidLine("expected two vertex ids");
        }
        try {
            NodeId a = std::stoi(tokens[0]);
            NodeId b = std::stoi(tokens[1]);
            graphEdges.push_back(std::make_pair(a, b));
            graphEdges.push_back(std::make_pair(b, a));
        } catch (const std::invalid_argument&) {
            throw invalidLine("invalid vertex id");
        } catch (const std::out_of_range&) {
            throw invalidLine("vertex id out of range");
        }
    }

    return Graph(graphEdges);
}

void GraphIO::writeToEdgeList(std::string filePath, const Graph& g) {
    std::ofstream fil(filePath);
    if (!fil.is_open()) {
        throw std::runtime_error("could not open " + filePath + " for writing");
    }
    for (int v = 0; v < g.getNumVertices(); v++) {
        for (int u : g.getNeighbors(v)) {
            if (v < u) {
                fil << v << " " << u << "\n";
            }
        }
    }
}
