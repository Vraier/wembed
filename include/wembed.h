#pragma once

#include <vector>
#include <set>
#include <string>
#include <map>
#include <memory>

namespace wembed {

#ifndef _WEMBED_IS_IMPL
namespace impl {
    // forward declaration
    class EmbeddingGraph;
    class EmbedderInterface;
}
#endif

using NodeId = int;
using EdgeId = int;

// forward declaration
class Embedder;

enum class SpatialIndex { RTree = 0, SNN = 1 };

struct Options {
    // Embedding parameters
    int embeddingDimension = 4;
    bool useWeights = true;                      // whether node weights are used for high-degree nodes
    bool layeredEmbedding = false;               // use multiple layers of contracted graphs to compute the embedding
    double dimensionHint = -1.0;                 // hint for the dimension of the input graph

    // Gradient descent parameters
    double coolingFactor = 0.99;                 // strong influence on runtime but increases quality
    double learningRate = 10;                    // learning rate
    int maxIterations = 1000;

    // Other parameters
    SpatialIndex indexType = SpatialIndex::SNN;  // spatial index used for the embedding (only affects running time)
};

class Graph {
   public:
    Graph(std::unique_ptr<impl::EmbeddingGraph>&& graph);
    ~Graph();

    Graph(const Graph& other) = delete;
    Graph& operator=(const Graph& other) = delete;
    Graph(Graph&& other);
    Graph& operator=(Graph&& other);

    // global information
    NodeId getNumVertices() const;
    EdgeId getNumEdges() const;

    // neighborhood information
    std::vector<EdgeId> getEdges(NodeId v) const;
    std::vector<NodeId> getNeighbors(NodeId v) const;
    int getNumNeighbors(NodeId v) const;
    NodeId getEdgeTarget(EdgeId e) const;
    bool areNeighbors(NodeId v, NodeId u) const;

    std::string toString() const;

   private:
    friend Embedder createEmbedder(const Graph& g, const Options& options);

    std::unique_ptr<impl::EmbeddingGraph> _graph;
};

class Embedder {
   public:
    Embedder(std::unique_ptr<impl::EmbedderInterface>&& embedder);
    ~Embedder();

    Embedder(const Embedder& other) = delete;
    Embedder& operator=(const Embedder& other) = delete;
    Embedder(Embedder&& other);
    Embedder& operator=(Embedder&& other);

    // embedding calculation
    void calculateStep();
    bool isFinished();
    void calculateEmbedding();

    // accessors
    Graph getCurrentGraph() const;
    std::vector<std::vector<double>> getCoordinates() const;
    std::vector<double> getWeights() const;
    void setCoordinates(const std::vector<std::vector<double>>& coordinates);
    void setWeights(const std::vector<double>& weights);

    void writeCoordinates(const std::string& filePath, bool writeWeights = false) const;

   private:
    std::unique_ptr<impl::EmbedderInterface> _embedder;
};

Embedder createEmbedder(const Graph& g, const Options& options);

Graph graphFromEdges(const std::vector<std::pair<NodeId, NodeId>>& edges);

Graph graphFromNeighborhoods(const std::map<NodeId, std::set<NodeId>>& nodeToNeighborhood);

Graph graphFromEdgeListFile(const std::string& filePath,
                            const std::string& comment = "#",
                            const std::string& delimiter = " ");

void setSeed(int seed);

}  // namespace wembed
