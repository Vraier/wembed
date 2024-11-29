#pragma once

#include "AbstractSimpleEmbedder.hpp"
#include "EmbedderOptions.hpp"
#include "WeightedRTree.hpp"

using RepellingCandidates = std::vector<std::pair<NodeId, NodeId>>;
using Timer = util::Timer;

class SamplingHeuristic {
   public:
    virtual ~SamplingHeuristic() {};

    /**
     * Returns a list of all pairs of nodes that a repulsion force should be calculated for
     */
    virtual RepellingCandidates calculateRepellingCandidates(const EmbeddedGraph& g, Timer& timer) = 0;

    /**
     * Returns the number of samples that should be used for a node
     * The samples will be proportional to the average degree or the degree of v
     */
    virtual int getNumSamplesForNode(const EmbeddedGraph& graph, NodeId v, int numSamples, bool uniformSampling);
};

/**
 * Euclidean embedder that ignores weights and tries to linearize the sigmoid forces
 */
class SimpleSamplingEmbedder : public AbstractSimpleEmbedder {
   public:
    SimpleSamplingEmbedder(Graph& g, EmbedderOptions opts)
        : AbstractSimpleEmbedder(g, opts), samplingHeuristic(createSamplingHeuristic(options.samplingType)) {};

   protected:
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u) override;
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u) override;

    virtual void calculateAllRepellingForces() override;
    virtual void dumpDebugAtTermination() override;

   private:
    std::unique_ptr<SamplingHeuristic> samplingHeuristic;

    // used to calculate sampling metrics
    int numEffectiveRepForceCalculations = 0;

    std::map<int, int> possibleRepForces;  // stores how many rep forces are found by quadratic
    std::map<int, int> correctRepForces;   // stores how many of the possible rep forces were found
    std::map<int, int> foundRepForces;     // stores how many candidates were found

    std::unique_ptr<SamplingHeuristic> createSamplingHeuristic(SamplingHeuristicType type);
    double getSimilarity(double norm, double wu, double wv);
};

/**
 * Calculates all possible repelling forces -> O(n^2) per step
 */
class QuadraticSampling : public SamplingHeuristic {
   public:
    QuadraticSampling() {};
    virtual RepellingCandidates calculateRepellingCandidates(const EmbeddedGraph& g, Timer& timer) override;
};

/**
 * Calculates rep forces for nSamples for every vertex -> O(nSamples * n)
 */
class RandomSampling : public SamplingHeuristic {
   public:
    RandomSampling(int numNegativeSamples, bool uniformSampling)
        : numNegativeSamples(numNegativeSamples), uniformSampling(uniformSampling) {};
    virtual RepellingCandidates calculateRepellingCandidates(const EmbeddedGraph& g, Timer& timer) override;

   private:
    int numNegativeSamples;
    bool uniformSampling;
};

/**
 * Generates a girg based on the current positions and weights.
 * The girg is used to determine close vertices and calculate repelling forces.
 */
class GirgGenSampling : public SamplingHeuristic {
   public:
    GirgGenSampling(int dimension, double averageDegree) : dimension(dimension), averageDegree(averageDegree) {};
    virtual RepellingCandidates calculateRepellingCandidates(const EmbeddedGraph& g, Timer& timer) override;

   private:
    int dimension;
    double averageDegree;
};

/**
 * Does BFS from every node until enough neighbors are found
 */
class BFSSampling : public SamplingHeuristic {
   public:
    BFSSampling(const EmbeddedGraph& g, int numSamples, bool uniformSampling);

    /**
     * Always returns the same candidates
     */
    virtual RepellingCandidates calculateRepellingCandidates(const EmbeddedGraph& g, Timer& timer) override;

   private:
    RepellingCandidates candidates;  // will be constant for every iteration
};

/**
 * Does dijkstra from every node until enough neighbors are found.
 * The edge costs are (deg(u)*deg(v))^(1/dim hint)
 */
class DistanceSampling : public SamplingHeuristic {
   public:
    DistanceSampling(const EmbeddedGraph& g, int numSamples, int dimHint, bool uniformSampling);

    /**
     * Always returns the same candidates
     */
    virtual RepellingCandidates calculateRepellingCandidates(const EmbeddedGraph& g, Timer& timer) override;

   private:
    RepellingCandidates candidates;  // will be constant for every iteration
};

class RTreeSampling : public SamplingHeuristic {
   public:
    RTreeSampling(const EmbeddedGraph& g, const std::vector<double>& weights, double edgeLength, double doublingFactor, bool useInfNorm)
        : weightBuckets(WeightedRTree::getDoublingWeightBuckets(weights, doublingFactor)),
          edgeLength(edgeLength),
          useInfNorm(useInfNorm),
          rtree(g.getDimension()) {
        //LOG_INFO("Using " << weightBuckets.size() + 1 << " weight class[es]");
        std::vector<std::pair<CVecRef, NodeId>> values;
        for (NodeId v = 0; v < g.getNumVertices(); v++) {
            values.push_back(std::make_pair(g.getPosition(v), v));
        }
        rtree.updateRTree(g.coordinates, g.getAllNodeWeights(), weightBuckets);
    };

    virtual RepellingCandidates calculateRepellingCandidates(const EmbeddedGraph& g, Timer& timer) override;

    virtual std::vector<NodeId> calculateRepellingCandidatesForNode(const EmbeddedGraph& g, NodeId v, Timer& timer) const;

   private:
    std::vector<double> weightBuckets;
    double edgeLength;
    bool useInfNorm = false;
    WeightedRTree rtree;
};
