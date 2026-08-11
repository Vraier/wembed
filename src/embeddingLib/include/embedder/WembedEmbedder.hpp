#pragma once

#include <mutex>

#include <memory>

#include "AdamOptimizer.hpp"
#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "Optimizer.hpp"
#include "SimpleOptimizer.hpp"
#include "VecList.hpp"

class WembedEmbedder : public EmbedderInterface {

    std::shared_ptr<util::Timer> timer;

    uint32_t numRepForceCalculations = 0;

    std::vector<double> invExpWeights;
    std::unique_ptr<Optimizer> posOptimizer;

    std::vector<std::mutex> candidateLocks;

    static std::unique_ptr<Optimizer> makePosOptimizer(const EmbedderOptions &opts, uint32_t numVertices) {
        switch (opts.optimizerType) {
            case OptimizerType::Simple:
                return std::make_unique<SimpleOptimizer>(opts.embeddingDimension, numVertices,
                                                         opts.learningRate, opts.coolingFactor,
                                                         opts.simpleOptMaxDisplacement);
            case OptimizerType::Adam:
                return std::make_unique<AdamOptimizer>(opts.embeddingDimension, numVertices,
                                                       opts.learningRate, opts.coolingFactor,
                                                       0.9, 0.999, 1e-8);
        }
        return std::make_unique<AdamOptimizer>(opts.embeddingDimension, numVertices,
                                               opts.learningRate, opts.coolingFactor,
                                               0.9, 0.999, 1e-8);
    }

    /**
     * Functions to compute the forces between two vertices
     */
    void calculateAllAttractingForces();
    void calculateAllRepellingForces();
    void calculateAllCentreForces();
    // Force functions return the loss contribution of this pair
    // so the callers can accumulate it
    double attractionForce(NodeId v, NodeId u, VecBuffer<1>& forceBuffer);
    double repellingForce(NodeId v, NodeId u, TmpVec<0>& result);
    double scatterRepulsion(NodeId v, const std::vector<NodeId>& candidates, VecList& forces, size_t threadCount);
    void applyGravityCentre();

    /**
     * Computes all nodes to do a repulsion force computation with node v
     */
    std::vector<NodeId> getRepellingCandidatesForNode(NodeId v, VecBuffer<2> &buffer) const;

    /**
     * Updates spacial data structure
     */
    void selectNodes(std::vector<std::pair<CVecRef, NodeId>>& points);
    void updateIndex();

    [[nodiscard]] std::vector<NodeId> sampleRandomNoise(int32_t numNodes) const;


    public:
    WembedEmbedder(const Graph& g,
                      const EmbedderOptions &opts,
                      const std::shared_ptr<util::Timer> &timer_ptr = std::make_shared<util::Timer>())
                      : EmbedderInterface(g, opts),
                        timer(timer_ptr),
                        invExpWeights(g.getNumVertices()),
                        posOptimizer(makePosOptimizer(opts, g.getNumVertices())),
                        candidateLocks(g.getNumVertices())
    {

        WembedEmbedder::setCoordinates(constructRandomCoordinates());

        switch (opts.weightType) {
            case WeightType::Degree:
                WembedEmbedder::setWeights(rescaleWeights(opts.dimensionHint,
                                                             opts.embeddingDimension,
                                                             constructDegreeWeights(g)));
                break;
            case WeightType::Unit:
                WembedEmbedder::setWeights(constructUnitWeights(graphSize()));
                break;
        }
    }

    virtual ~WembedEmbedder() override = default;
    WembedEmbedder(WembedEmbedder&&) = default;
    WembedEmbedder& operator=(WembedEmbedder&&) = default;
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
