#pragma once

#include <vector>

#include "EmbedderOptions.hpp"
#include "EmbedderState.hpp"
#include "Graph.hpp"
#include "Timings.hpp"
#include "VecList.hpp"

/**
 * Loss values from the last completed force computation.
 */
struct EmbeddingLoss {
    float attractive;
    float repulsive;
    float total;
};

/**
 * Interface for weighted embedder classes.
 */
class EmbedderInterface {

    protected:
    Graph graph;            // the (current) graph being embedded
    EmbedderOptions opts;   // configuration
    EmbedderState state;    // all mutable run state (layout, buffers, last-step observables)

    EmbedderInterface(const Graph& g, const EmbedderOptions& opts)
                        : graph(g),
                          opts(opts),
                          state(g.getNumVertices(), opts.embeddingDimension, opts.indexType)
    {
        state.lastLearningRate = opts.learningRate;
    }

    /**
     * Returns the number of vertices of the graph
     */
    [[nodiscard]] inline uint32_t graphSize() const {
        return this->graph.getNumVertices();
    }

    /**
     * Sorts the node IDs according to the nodes weight
     */
    void sortNodes() {
        std::iota(state.sortedNodeIDs.begin(), state.sortedNodeIDs.end(), 0);
        std::ranges::sort(state.sortedNodeIDs,
                          [this](const int a , const int b) -> bool {return this->state.currentWeights[a] > this->state.currentWeights[b];});
    }


    // embedding functions
    /**
     * randomly places the nodes in the embedding space
     * @return A vector of coordinates, where vector[v] are the coordinates of the node with ID v
     */
    [[nodiscard]] std::vector<std::vector<float>> constructRandomCoordinates() const {
        const int32_t dimension = this->opts.embeddingDimension;
        const float CUBE_SIDE_LENGTH = Toolkit::myPowf(static_cast<float>(graphSize()), 1.0f / static_cast<float>(dimension));
        return Rand::randomCoordinatesf(static_cast<int>(graphSize()), dimension, CUBE_SIDE_LENGTH);
    }


   public:
    virtual ~EmbedderInterface() = default;
    EmbedderInterface(const EmbedderInterface&) = delete;
    EmbedderInterface& operator=(const EmbedderInterface&) = delete;
    EmbedderInterface(EmbedderInterface&&) = default;
    EmbedderInterface& operator=(EmbedderInterface&&) = default;

    /**
     * Number of vertices in the current graph the embedder is operating on.
     * For LayeredEmbedder, this changes across coarsening layers.
     */
    virtual int getNumVertices() const {
        return static_cast<int>(this->state.currentPositions.size());
    }

    /**
     * Dimension of the embedding space.
     */
    virtual int getEmbeddingDimension() const {
        return static_cast<int>(this->state.currentPositions.dimension());
    }

    /**
     * Copy coordinates row-major into a caller-owned buffer of at least
     * getNumVertices() * getEmbeddingDimension() floats. Zero allocation.
     */
    virtual void copyCoordinatesTo(float* out) const {
        this->state.currentPositions.copyToFlat(out);
    }

    /**
     * Loss from the most recent force computation
     */
    virtual EmbeddingLoss getLoss() const {
        return {this->state.lastAttractLoss,
                this->state.lastRepelLoss,
                this->state.lastAttractLoss + this->state.lastRepelLoss};
    }

    /**
     * Learning rate the optimizer used in the most recent step
     * (before the first step: the initial learning rate).
     */
    virtual float getCurrentLearningRate() const {
        return this->state.lastLearningRate;
    }

    /**
     * This is the signal the displacement stopping criterion watches.
     */
    virtual float getLastRelDisplacement() const {
        return this->state.lastRelDisplacement;
    }

    /**
     * This is the signal the loss stopping criterion watches.
     */
    virtual float getLastRelLossImprovement() const {
        return this->state.lastRelLossImprovement;
    }

    /**
     * Advances the embedding by a single gradient descent step.
     */
    virtual void calculateStep() = 0;

    /**
     * Returns whether the embedder is finished (enough steps or insignificant change).
     */
    virtual bool isFinished() = 0;

    /**
     * Calculates the whole embedding until termination criterion is met.
     */
    virtual void calculateEmbedding() = 0;

    /**
     * Returns the current graph. Manly important for layered embedder
     */
    virtual Graph getCurrentGraph() = 0;

    /**
     * Returns the current coordinates of the nodes.
     */
    virtual std::vector<std::vector<float>> getCoordinates() = 0;

    /**
     * Returns the current weights of the nodes.
     */
    virtual std::vector<float> getWeights() = 0;

    /*
     * Returns timing results for the duration of different phases of the embedding
     */
    virtual std::vector<util::TimingResult> getTimings() = 0;

    /**
     * Sets the coordinates of the nodes.
     * Can be used to set initial coordinates.
     */
    virtual void setCoordinates(const std::vector<std::vector<float>> &coordinates) = 0;

    /**
     * Sets the weights of the nodes.
     * Can be used to set initial weights.
     */
    virtual void setWeights(const std::vector<float> &weights) = 0;
};