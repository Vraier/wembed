#pragma once

#include <vector>

#include "Graph.hpp"
#include "Timings.hpp"

/**
 * Interface for weighted embedder classes.
 */
class EmbedderInterface {
   public:
    virtual ~EmbedderInterface() {};

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