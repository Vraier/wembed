#pragma once

/**
 * Interface for embedders that make use of a graph hierarchy.
 */
class WeightedApproxEmbedder {
   public:
    virtual ~WeightedApproxEmbedder(){};

    virtual void initializeNewRun() = 0;
    virtual void calculateStep() = 0;
    virtual void calculateEmbedding() = 0;
    virtual bool isFinished() = 0;
};