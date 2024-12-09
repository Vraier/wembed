#pragma once

#include <memory>

#include "EmbeddedGraph.hpp"
#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "Graph.hpp"
#include "Optimizer.hpp"
#include "Timings.hpp"
#include "VecList.hpp"

/**
 * Abstract class that represents a generic force directed algorithm.
 * The concrete attracting and repelling forces have to be implemented by the inheriting class.
 * Many other methods contain sensible default implementations.
 *
 * All pair repulsion forces are considered, usually resulting in O(n^2) runtime.
 * The sampling version can improve on this by only considering a small set of nodes or using geometrical data
 * structures.
 */
class AbstractSimpleEmbedder : public EmbedderInterface {
    using Timer = util::Timer;

   public:
    AbstractSimpleEmbedder(Graph& g, EmbedderOptions opts)
        : buffer(opts.embeddingDimension),
          graph(opts.embeddingDimension, g),
          options(opts),
          currentForce(opts.embeddingDimension, g.getNumVertices()),
          oldPositions(opts.embeddingDimension, g.getNumVertices()),
          oldWeights(g.getNumVertices()),
          newWeights(g.getNumVertices()),
          currentWeightForce(g.getNumVertices()) {
        setWeights(AbstractSimpleEmbedder::rescaleWeights(opts.dimensionHint, opts.embeddingDimension,
                                                        constructDegreeWeights(g)));
        setCoordinates(constructRandomCoordinates(opts.embeddingDimension, g.getNumVertices()));
        initializeOptimizer();
    };

    virtual ~AbstractSimpleEmbedder() {};

    virtual void calculateStep();
    virtual void calculateEmbedding();

    virtual Graph getCurrentGraph();
    virtual std::vector<std::vector<double>> getCoordinates();
    virtual std::vector<double> getWeights();

    virtual void setCoordinates(const std::vector<std::vector<double>>& coordinates);
    virtual void setWeights(const std::vector<double>& weights);

    virtual VecList getCurrentLayout();
    virtual EmbeddedGraph getEmbeddedGraph();
    virtual bool isFinished();

    virtual void dumpDebugAtTermination();

    std::vector<util::TimingResult> getTimings() const;


    /**
     * Helper functions to construct initial embedding
     */

    static std::vector<double> constructDegreeWeights(const Graph& g);
    static std::vector<double> constructUnitWeights(int N);
    /**
     * Rescales the weights to have average weight 1 and use dimension hint.
     */
    static std::vector<double> rescaleWeights(int dimensionHint, int embeddingDimension,
                                              const std::vector<double>& weights);

    static std::vector<std::vector<double>> constructRandomCoordinates(int dimension, int N);

   protected:
    Timer timer;
    VecBuffer<20> buffer;

    static constexpr int REP_BUFFER = 2;
    static constexpr int ATTR_BUFFER = 3;
    static constexpr int FORCE_STEP_BUFFER = 4;
    static constexpr int WEIGHT_ATTR_BUFFER = 5;
    static constexpr int WEIGHT_REP_BUFFER = 6;

    virtual void calculateForceStep();

    /**
     * Repulsion force that pushes v away from u
     */
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u) = 0;

    /**
     * Attracting force that pulls v towards u
     */
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u) = 0;

    /**
     * Adds repulsion forces to the current force vector.
     * The repulsion forces should be set to zero by the calculateForceStep method.
     */
    void calculateAllAttractingForces();

    /**
     * Same as calculateAllAttractingForces.
     *
     * Also selects the right sampling method for calculating repelling forces.
     */
    virtual void calculateAllRepellingForces();

    double sumWeightRepulsionForce(NodeId v);
    double sumWeightAttractionForce(NodeId v);

    virtual void calculateWeightStep();

    /**
     * force that reduces vs weight based on the non neighbor u
     */
    virtual double weightRepulsionForce(NodeId v, NodeId u);

    /**
     * force that increases vs weight based on the neighbor u
     */
    virtual double weightAttractionForce(NodeId v, NodeId u);

    /**
     * Initializes the optimizer to use naive or Adam optimizer
     */
    virtual void initializeOptimizer();

    EmbeddedGraph graph;
    EmbedderOptions options;
    std::unique_ptr<Optimizer> optimizer;

    int currIteration = 0;
    bool insignificantPosChange = false;
    bool insignificantWeightChange = false;
    int numForceSteps = 0;
    int numWeightSteps = 0;

    VecList currentForce;
    VecList oldPositions;
    std::vector<double> oldWeights;
    std::vector<double> newWeights;
    std::vector<double> currentWeightForce;

    std::vector<std::vector<double>> initialCoords;
    std::vector<double> initialWeights;
};
