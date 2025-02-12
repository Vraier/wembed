#pragma once

#include "Graph.hpp"
#include "Options.hpp"

/**
 * this is basically a wrapper around the girg library.
 * we can convince the library to output girgs and euclidean graphs.
 * even if they are not on the torus.
 */
class GirgGenerator {
   public:
    /**
     * Generates a girg with the given parameters. Only returns the largest connected component.
     * Also returns the coordinates for every node and the weights.
     */
    static std::tuple<Graph, std::vector<std::vector<double>>, std::vector<double>> generateRandomGraph(
        Options options);
};
