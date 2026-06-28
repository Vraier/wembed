#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace wembed {

#ifndef _WEMBED_IS_IMPL
namespace impl {
    // forward declaration
    class EmbeddingGraph;
    class EmbedderInterface;
}
#endif

using NodeId = int32_t;
using EdgeId = int32_t;

// forward declaration
class Embedder;

enum SpatialIndex : int32_t {
    IndexSNN = 1,
    IndexSprk = 2,
};

// edge type. Used in place of std::pair<NodeId, NodeId>.
struct Edge {
    NodeId src;
    NodeId dst;
};

// A single entry in the embedder's timing breakdown. depth indicates
// nesting level (0 = top-level phase). value is wall-clock time in seconds.
struct TimingResult {
    uint64_t depth;
    std::string displayName;
    double value;
};

struct Options {
    // Embedding parameters
    int32_t embeddingDimension = 4;
    bool useWeights = true;                      // true: degree-based weights; false: unit weights
    double dimensionHint = -1.0;                 // hint for the dimension of the input graph (-1 = auto)
    bool layeredEmbedding = false;               // multilevel embedding via graph coarsening

    // Force parameters
    SpatialIndex indexType = IndexSprk;
    double attractionScale = 1.0;
    double repulsionScale = 1.0;
    double edgeLength = 1.0;
    double expansionStretch = 1.0;               // stretch applied during layer expansion

    // Gradient descent parameters
    double coolingFactor = 0.99;                 // lower = faster cooldown
    double learningRate = 10.0;
    int32_t maxIterations = 1000;
    double positionMinChange = 1e-4;             // halt threshold on position change
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

    // Returns the full undirected edge list. Each edge appears exactly once with src < dst.
    // Length equals getNumEdges().
    std::vector<Edge> getEdgeList() const;

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
    // TODO: revisit these for efficiency. let caller allocate memory
    Graph getCurrentGraph() const;
    std::vector<std::vector<double>> getCoordinates() const;
    std::vector<double> getWeights() const;
    void setCoordinates(const std::vector<std::vector<double>>& coordinates);
    void setWeights(const std::vector<double>& weights);

    // Hierarchical breakdown of time spent in each phase of the embedding.
    std::vector<TimingResult> getTimings() const;

    void writeCoordinates(const std::string& filePath, bool writeWeights = true) const;

   private:
    std::unique_ptr<impl::EmbedderInterface> _embedder;
};

Embedder createEmbedder(const Graph& g, const Options& options);

// Build a graph from an edge list. Each undirected edge should appear exactly once.
// Vertex IDs must be consecutive starting at 0.
Graph graphFromEdges(const std::vector<Edge>& edges);

Graph graphFromEdgeListFile(const std::string& filePath,
                            const std::string& comment = "#",
                            const std::string& delimiter = " ");

// Read a coordinate file (one row per vertex). Useful for resuming an embedding
// via Embedder::setCoordinates.
std::vector<std::vector<double>> readCoordinatesFromFile(
    const std::string& filePath,
    const std::string& comment = "%",
    const std::string& delimiter = ",");

// Pretty-print a hierarchical timing breakdown.
std::string timingsToString(const std::vector<TimingResult>& timings);

void setSeed(int seed);

}  // namespace wembed
