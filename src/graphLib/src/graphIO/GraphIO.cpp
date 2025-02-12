#include "GraphIO.hpp"

#include <fstream>
#include <iostream>
#include <regex>

#include "Macros.hpp"
#include "StringManipulation.hpp"

Graph GraphIO::readEdgeList(std::string filePath, std::string comment, std::string delimiter) {
    LOG_INFO("Reading in graph from edge list");

    std::vector<std::pair<NodeId, NodeId>> graphEdges;

    std::ifstream input(filePath);

    if (!input.good()) {
        LOG_ERROR("Could not find file: " << filePath);
        return Graph(graphEdges);
    }

    std::string line;
    while (std::getline(input, line)) {
        if (util::startsWith(line, comment)) {
            // line starts with comment
            continue;
        }
        std::vector<std::string> tokens = util::splitIntoTokens(line, delimiter);
        if (tokens.size() < 2) {
            LOG_ERROR("Invalid line format: " << line);
            continue;
        }

        try {
            NodeId a = std::stoi(tokens[0]);
            NodeId b = std::stoi(tokens[1]);
            graphEdges.push_back(std::make_pair(a, b));
            graphEdges.push_back(std::make_pair(b, a));
        } catch (const std::invalid_argument& e) {
            LOG_ERROR("Invalid token in line: " << line);
            continue;
        } catch (const std::out_of_range& e) {
            LOG_ERROR("Token out of range in line: " << line);
            continue;
        }
    }
    input.close();

    Graph g(graphEdges);
    return g;
}

void GraphIO::writeToEdgeList(std::string filePath, const Graph& g) {
    std::ofstream fil;
    fil.open(filePath);
    for (int v = 0; v < g.getNumVertices(); v++) {
        for (int u : g.getNeighbors(v)) {
            if (v < u) {
                fil << v << " " << u << "\n";
            }
        }
    }
    fil.close();
}

Graph GraphIO::readBipartiteEdgeList(std::string filePath, std::string comment, std::string delimiter) {
    LOG_INFO("Reading in bipartite graph from edge list " << filePath);

    std::vector<std::pair<NodeId, NodeId>> graphEdges;

    std::ifstream input(filePath);
    if (!input.good()) {
        LOG_ERROR("Could not find file: " << filePath);
        return Graph(graphEdges);
    }

    std::string line;
    std::getline(input, line);
    std::vector<std::string> tokens = util::splitIntoTokens(line, delimiter);
    if (tokens.size() != 3) {
        LOG_ERROR("Invalid first line format: " << line);
        return Graph(graphEdges);
    }

    std::string prefix = tokens[0];
    int a = std::stoi(tokens[1]);
    int b = std::stoi(tokens[2]);
    ASSERT(prefix == "#psizes", "Invalid prefix in bipartite edge list");

    while (std::getline(input, line)) {
        if (util::startsWith(line, comment)) {
            // line starts with comment
            continue;
        }
        std::vector<std::string> tokens = util::splitIntoTokens(line, delimiter);
        if (tokens.size() != 2) {
            LOG_ERROR("Invalid line format: " << line);
            continue;
        }

        try {
            NodeId u = std::stoi(tokens[0]);
            NodeId v = std::stoi(tokens[1]);
            graphEdges.push_back(std::make_pair(u, v));
        } catch (const std::invalid_argument& e) {
            LOG_ERROR("Invalid token in line: " << line);
            continue;
        } catch (const std::out_of_range& e) {
            LOG_ERROR("Token out of range in line: " << line);
            continue;
        }
    }
    input.close();

    Graph g(graphEdges);
    ASSERT(a+b == g.getNumVertices(), "Number of vertices does not match the partition sizes");

    // set up the colors
    std::vector<int> colors(g.getNumVertices());
    for (int i = 0; i < g.getNumVertices(); i++) {
        colors[i] = i < a ? 0 : 1;
    }
    g.setColors(colors);

    return g;
}
