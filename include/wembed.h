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

enum StopCriterion : int32_t {
    StopDisplacement = 0,  // stop when the relative per-step node movement settles
    StopLoss = 1,          // stop when the (smoothed) loss stagnates (default)
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
    float value;
};

// Loss values from the most recent embedding step
struct Loss {
    float attractive;
    float repulsive;
    float total;
};

struct Options {
    // Embedding parameters
    int32_t embeddingDimension = 4;
    bool useUnitWeights = false;                      // true: degree-based weights; false: unit weights
    float dimensionHint = -1.0;                 // hint for the dimension of the input graph (-1 = auto)
    bool layeredEmbedding = false;               // multilevel embedding via graph coarsening

    // Force parameters
    SpatialIndex indexType = IndexSprk;
    float attractionScale = 1.0;
    float repulsionScale = 1.0;
    float centreScale = 0.0;                    // pull toward origin; nonzero enables it (useful for unconnected graphs)
    float edgeLength = 1.0;
    float expansionStretch = 1.0;               // stretch applied during layer expansion

    // Gradient descent parameters
    OptimizerType optimizerType = OptimizerAdam;
    int32_t maxIterations = 10000;
    float simpleOptMaxDisplacement = 1.0;       // per-step cap (only used when optimizerType == OptimizerSimple)

    // Learning rate schedule (lr* prefix). Every parameter states which schedules read it.
    LRSchedule lrSchedule = LRExponentialCooling;
    float learningRate = 10.0;                  // initial learning rate (both schedules)
    int32_t warmupSteps = 20;                    // linear LR ramp-up over the first steps (both schedules)
    float lrCoolingFactor = 0.995;              // per-step multiplicative decay, lower = faster cooldown (LRExponentialCooling only)
    float lrDecayFactor = 0.5;                  // multiplicative drop on a decay event (LRLossAdaptive only)
    float lrDecayThreshold = 1e-2;              // decay when the loss-decrease rate stays below this (LRLossAdaptive only)
    int32_t lrAdaptPatience = 20;                // consecutive in-zone steps before a decay OR growth event (LRLossAdaptive only)
    float lrGrowthFactor = 1.0;                 // multiplicative growth while the loss keeps decreasing fast
                                                 // (LRLossAdaptive only; 1.0 disables growth -> pure plateau decay)
    float lrGrowthThreshold = 1e-1;             // grow when the loss-decrease rate stays above this (LRLossAdaptive only)

    // Stopping criterion (maxIterations always applies as a hard cap).
    StopCriterion stopCriterion = StopLoss;  // which signal terminates the run

    // Displacement stopping criterion (StopDisplacement).
    float stopDisplacementTol = 3e-4;           // relative per-step node movement (mean displacement / radius of
                                                 // gyration) below which the layout counts as settled
    int32_t stopDisplacementPatience = 5;        // settled steps in a row before stopping

    // Shared loss-progress signal (loss* prefix): windowed relative loss decrease rate(t),
    // read by both the loss stop criterion and the LRLossAdaptive schedule.
    float lossSmoothingFactor = 0.3;            // EMA weight of the newest loss sample before the monitor sees it
                                                 // (1.0 disables smoothing); a light denoise on rate(t)
    int32_t lossRateWindow = 30;                 // steps over which the relative loss-decrease rate is measured

    // Loss stagnation stopping criterion (StopLoss).
    float stopLossTol = 1e-3;                   // ftol: converged once rate(t) stays below this (relative decrease over window)
    int32_t stopLossPatience = 50;               // sub-tolerance steps in a row before stopping
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

    // flat copy of coordinates getNumVertices() * getEmbeddingDimension() floats, row-major.
    void copyCoordinatesTo(float* out) const;

    Graph getCurrentGraph() const;
    std::vector<std::vector<float>> getCoordinates() const;
    std::vector<float> getWeights() const;
    void setCoordinates(const std::vector<std::vector<float>>& coordinates);
    void setWeights(const std::vector<float>& weights);

    // Hierarchical breakdown of time spent in each phase of the embedding.
    std::vector<TimingResult> getTimings() const;

    // Loss from the most recent step.
    Loss getLoss() const;

    // Learning rate the optimizer used in the most recent step
    // (before the first step: the initial learning rate).
    float getCurrentLearningRate() const;

    // Relative node displacement of the most recent step
    // (mean per-node movement / radius of gyration); the displacement stop watches this.
    float getLastRelDisplacement() const;

    // Windowed relative loss-decrease rate of the most recent step (rate(t));
    // the loss stop watches this (a step is stagnant when it stays below stopLossTol).
    float getLastRelLossImprovement() const;

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
std::vector<std::vector<float>> readCoordinatesFromFile(
    const std::string& filePath,
    const std::string& comment = "%",
    const std::string& delimiter = ",");

// Pretty-print a hierarchical timing breakdown.
std::string timingsToString(const std::vector<TimingResult>& timings);

void setSeed(int seed);

}  // namespace wembed
