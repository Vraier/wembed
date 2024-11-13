#pragma once

#include "EmbeddedGraph.hpp"
#include "Graph.hpp"
#include "GraphHierarchy.hpp"
#include "EmbOptions.hpp"
#include "WeightedApproxEmbedder.hpp"

/**
 * Coarsens the graph to build a hierarchy of smaller graphs
 * Uses WeightedSpring/WeightedStress embedder to embed the single hierarchies
*/
class HierarchyEmbedder {
   public:
    HierarchyEmbedder(OptionValues options, const Graph &g) : opts(options),
                                                              originalGraph(g),
                                                              initialWeights(g.getNumVertices(), 1.0),
                                                              buffer(options.dimension){};
    ~HierarchyEmbedder() {
        delete graphHierarchy;
    }

    void setInitialWeights(const std::vector<double> &weights);
    void initializeNewRun();
    void calculateEmbedding();
    EmbeddedGraph getEmbeddedGraph();

   private:
    GraphHierarchy *buildHierarchy();  // chooses the correct partitioning algorithm
    void expandPositions(int currLayer);
    WeightedApproxEmbedder *getEmbedderForLayer(int currLayer);

    OptionValues opts;
    Graph originalGraph;
    std::vector<double> initialWeights;

    GraphHierarchy *graphHierarchy;

    VecBuffer<5> buffer;
};