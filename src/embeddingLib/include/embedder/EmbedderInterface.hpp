#pragma once

#include <vector>
#include <boost/geometry/core/coordinate_dimension.hpp>

#include "EmbedderOptions.hpp"
#include "EmbedderParameters.hpp"
#include "Graph.hpp"
#include "Timings.hpp"
#include "VecList.hpp"
#include "WeightedIndex.hpp"

/**
 * Interface for weighted embedder classes.
 */
class EmbedderInterface {

    protected:
    // graph information
    Graph graph;
    VecList currentPositions;
    std::vector<double> currentWeights;
    std::vector<int32_t> sortedNodeIDs;

    // embedding information
    EmbedderOptions opts;
    EmbedderParameters params;

    EmbedderInterface(const Graph& g, const EmbedderOptions& opts)
                        : graph(g),
                          currentPositions(opts.embeddingDimension, g.getNumVertices()),
                          currentWeights(g.getNumVertices()),
                          sortedNodeIDs(g.getNumVertices()),
                          opts(opts),
                          params(g.getNumVertices(), opts.embeddingDimension, opts.indexType)
    {

    }

    // graph functions
    [[nodiscard]] constexpr uint32_t graphSize() const {
        return this->graph.getNumVertices();
    }

    void sortNodes() {
        std::iota(sortedNodeIDs.begin(), sortedNodeIDs.end(), 0);
        std::ranges::sort(sortedNodeIDs,
                          [this](const int a , const int b) -> bool {return this->currentWeights[a] > this->currentWeights[b];});
    }


    // embedding functions
    // TODO: If only the (New)WembedEmbedder needs this, this might actually better be in another class
    // TODO: Alternatively, this could be in the random class
    [[nodiscard]] std::vector<std::vector<double>> constructRandomCoordinates() const {
        const int32_t dimension = this->opts.embeddingDimension;
        const double CUBE_SIDE_LENGTH = Toolkit::myPow(static_cast<float>(graphSize()), 1.0 / dimension);
        return Rand::randomCoordinates(graphSize(), dimension, CUBE_SIDE_LENGTH);
    }


   public:
    virtual ~EmbedderInterface() = default;

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
    virtual std::vector<std::vector<double>> getCoordinates() = 0;

    /**
     * Returns the current weights of the nodes.
     */
    virtual std::vector<double> getWeights() = 0;

    /*
     * Returns timing results for the duration of different phases of the embedding
     */
    virtual std::vector<util::TimingResult> getTimings() = 0;

    /**
     * Sets the coordinates of the nodes.
     * Can be used to set initial coordinates.
     */
    virtual void setCoordinates(const std::vector<std::vector<double>> &coordinates) = 0;

    /**
     * Sets the weights of the nodes.
     * Can be used to set initial weights.
     */
    virtual void setWeights(const std::vector<double> &weights) = 0;
};