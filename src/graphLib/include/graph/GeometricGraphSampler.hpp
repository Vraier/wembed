#pragma once

#include "Graph.hpp"

typedef std::pair<Graph, std::vector<std::vector<double>>> GraphCoordinatesPair;

class GeometricGraphSampler {
   public:
    /**
     * Generates a random disc with coordinates intersection graph with n nodes.
     * It automatically determins a good gird size and radius.
     * It only returns the largest connected component,
     * which can be smaller than n.
     *
     * Has quadratic runtime in n.
     */
    GraphCoordinatesPair generateRandomGraphWithCoordinates(int n);

    /**
     * Generates a random disc intersection graph with n nodes.
     * It automatically determins a good gird size and radius.
     * It only returns the largest connected component,
     * which can be smaller than n.
     *
     * Has quadratic runtime in n.
     *
     */
    Graph generateRandomGraph(int n);

   private:
    GraphCoordinatesPair generateRandomGraph(int n, double gridSize, double radius);
    GraphCoordinatesPair findLargesConnectedComponent(GraphCoordinatesPair &graphCoords);
};