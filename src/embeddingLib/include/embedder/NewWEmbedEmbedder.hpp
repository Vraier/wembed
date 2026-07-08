#pragma once

#include "AdamOptimizer.hpp"
#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "VecList.hpp"
#include "WeightedIndex.hpp"

class NewWEmbedEmbedder : public EmbedderInterface {

    std::shared_ptr<util::Timer> timer;

    uint32_t numRepForceCalculations = 0;

    std::vector<double> invExpWeights;
    AdamOptimizer posOptimizer;
    AdamOptimizer weightOptimizer;

    void debug_dumpWeights() const;

    /**
     * Functions to compute the forces between two vertices
     */
    void calculateAllAttractingForces();
    void calculateAllRepellingForces();
    void calculateAllCentreForces();
    void attractionForce(NodeId v, NodeId u, VecBuffer<1>& forceBuffer);
    void repellingForce(NodeId v, NodeId u, VecBuffer<1>& forceBuffer);
    void applyGravityCentre();

    /**
     * Computes all nodes to do a repulsion force computation with node v
     */
    std::vector<NodeId> getRepellingCandidatesForNode(NodeId v, VecBuffer<2> &buffer) const;

    /**
     * Updates spacial data structure
     */
    void updateIndex();

    [[nodiscard]] std::vector<NodeId> sampleRandomNoise(int32_t numNodes) const;


    public:
    NewWEmbedEmbedder(const Graph& g,
                      const EmbedderOptions &opts,
                      const std::shared_ptr<util::Timer> &timer_ptr = std::make_shared<util::Timer>())
                      : EmbedderInterface(g, opts),
                        timer(timer_ptr),
                        invExpWeights(g.getNumVertices()),
                        posOptimizer(opts.embeddingDimension, g.getNumVertices(), opts.learningRate, opts.coolingFactor, 0.9, 0.999, 1e-8),
                        weightOptimizer(opts.embeddingDimension, g.getNumVertices(), opts.weightLearningRate,opts.coolingFactor, 0.9, 0.999, 1e-8)
    {

        NewWEmbedEmbedder::setCoordinates(constructRandomCoordinates());

        switch (opts.weightType) {
            case WeightType::Degree:
                NewWEmbedEmbedder::setWeights(rescaleWeights(opts.dimensionHint,
                                                             opts.embeddingDimension,
                                                             constructDegreeWeights(g)));
                break;
            case WeightType::Unit:
                NewWEmbedEmbedder::setWeights(constructUnitWeights(graphSize()));
                break;
            default:
                LOG_ERROR("Weight type not supported");
        }
        if (opts.weightLearningRate > 0) {
            LOG_WARNING("There is no weight learning for this type of embedder");
        }
        if (opts.lpNorm != 2) {
            LOG_WARNING("Currently lpNorm = 2 is the only supported lpNorm");
        }
    }

    //TODO: Add Python Bindings?
    virtual ~NewWEmbedEmbedder() override = default;
    virtual void calculateStep() override;
    virtual bool isFinished() override;
    virtual void calculateEmbedding() override;
    virtual Graph getCurrentGraph() override;
    virtual std::vector<std::vector<double>> getCoordinates() override;
    virtual std::vector<double> getWeights() override;
    virtual std::vector<util::TimingResult> getTimings() override;
    virtual void setCoordinates(const std::vector<std::vector<double>> &coordinates) override;
    virtual void setWeights(const std::vector<double>& weights) override;

    [[nodiscard]] static std::vector<double> rescaleWeights(double dimensionHint, double embeddingDimension,
                                                        const std::vector<double>& weights);
    [[nodiscard]] static std::vector<double> constructDegreeWeights(const Graph& g);
    [[nodiscard]] static std::vector<double> constructUnitWeights(int N);
};
