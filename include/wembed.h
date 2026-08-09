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

enum OptimizerType : int32_t {
    OptimizerSimple = 0,
    OptimizerAdam = 1,
};

enum LRSchedule : int32_t {
    LRExponentialCooling = 0,
    LRLossAdaptive = 1,
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

// Loss values from the most recent embedding step
struct Loss {
    double attractive;
    double repulsive;
    double total;
};

struct Options {
    // Embedding parameters
    int32_t embeddingDimension = 4;
    bool useUnitWeights = false;                      // true: degree-based weights; false: unit weights
    double dimensionHint = -1.0;                 // hint for the dimension of the input graph (-1 = auto)
    bool layeredEmbedding = false;               // multilevel embedding via graph coarsening

    // Force parameters
    SpatialIndex indexType = IndexSprk;
    double attractionScale = 1.0;
    double repulsionScale = 1.0;
    double centreScale = 0.0;                    // pull toward origin; nonzero enables it (useful for unconnected graphs)
    double edgeLength = 1.0;
    double expansionStretch = 1.0;               // stretch applied during layer expansion

    // Gradient descent parameters
    OptimizerType optimizerType = OptimizerAdam;
    int32_t maxIterations = 1000;
    double simpleOptMaxDisplacement = 1.0;       // per-step cap (only used when optimizerType == OptimizerSimple)

    // Learning rate schedule. Every parameter states which schedules read it.
    LRSchedule lrSchedule = LRExponentialCooling;
    double learningRate = 10.0;                  // initial learning rate (both schedules)
    int32_t warmupSteps = 0;                     // linear LR ramp-up over the first steps (both schedules)
    double coolingFactor = 0.99;                 // per-step multiplicative decay, lower = faster cooldown (LRExponentialCooling only)
    double decayFactor = 0.5;                    // multiplicative drop on a decay event (LRLossAdaptive only)
    int32_t plateauPatience = 10;                // stagnant steps in a row before a decay event (LRLossAdaptive only)
    double growthFactor = 1.05;                  // multiplicative growth after a significant new best loss
                                                 // (LRLossAdaptive only; 1.0 disables growth)
    double growthRelTol = 3e-2;                  // relative loss improvement over the best-so-far that triggers
                                                 // an LR increase (LRLossAdaptive only)

    // Loss stagnation stopping criterion (maxIterations always applies as a hard cap).
    double stopRelTol = 3e-2;                    // relative loss improvement below which a step counts as stagnant
                                                 // (also feeds the stagnation counter that times LRLossAdaptive's decay events)
    int32_t stopPatience = 50;                   // stagnant steps in a row before stopping.
                                                 // When combined with LRLossAdaptive keep this >= 3 * plateauPatience
                                                 // so the schedule gets a few decay events before the embedding stops.
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
    bool isFinished() const;
    void calculateEmbedding();

    // accessors
    // Size accessors: reflect the CURRENT graph the embedder operates on
    // (for LayeredEmbedder, this changes across coarsening layers).
    int32_t getNumVertices() const;
    int32_t getEmbeddingDimension() const;

    // flat copy of coordinates getNumVertices() * getEmbeddingDimension() doubles, row-major.
    void copyCoordinatesTo(double* out) const;

    Graph getCurrentGraph() const;
    std::vector<std::vector<double>> getCoordinates() const;
    std::vector<double> getWeights() const;
    void setCoordinates(const std::vector<std::vector<double>>& coordinates);
    void setWeights(const std::vector<double>& weights);

    // Hierarchical breakdown of time spent in each phase of the embedding.
    std::vector<TimingResult> getTimings() const;

    // Loss from the most recent step.
    Loss getLoss() const;

    // Learning rate the optimizer used in the most recent step
    // (before the first step: the initial learning rate).
    double getCurrentLearningRate() const;

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
