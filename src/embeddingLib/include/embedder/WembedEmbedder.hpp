#pragma once

#include <memory>

#include "AdamOptimizer.hpp"
#include "ConvergenceMonitor.hpp"
#include "DisplacementMonitor.hpp"
#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "LRScheduler.hpp"
#include "Optimizer.hpp"
#include "SimpleOptimizer.hpp"
#include "VecList.hpp"
#include "WeightedIndex.hpp"

class WembedEmbedder : public EmbedderInterface {

    std::shared_ptr<util::Timer> timer;

    uint32_t numRepForceCalculations = 0;

    std::vector<double> invExpWeights;
    // per-node loss contribution of the last force computation; each node is
    // written by exactly one thread, then reduced deterministically (so the
    // stopping-criterion signal does not depend on thread count)
    std::vector<double> lossPerNode;
    // positions at the start of the current step and the per-node displacement /
    // squared radius derived from them; written one-thread-per-node then reduced
    // deterministically, exactly like lossPerNode
    VecList previousPositions;
    std::vector<double> perNodeDisplacement;
    std::vector<double> perNodeRadiusSq;
    std::unique_ptr<Optimizer> posOptimizer;
    // heap-owned and declared before the scheduler: LossAdaptive holds a reference to the monitor,
    // which stays valid when the embedder is moved (LayeredEmbedder moves it on layer expansion)
    std::unique_ptr<ConvergenceMonitor> convergenceMonitor;
    std::unique_ptr<DisplacementMonitor> displacementMonitor;
    std::unique_ptr<LRScheduler> lrScheduler;

    static std::unique_ptr<Optimizer> makePosOptimizer(const EmbedderOptions &opts, uint32_t numVertices) {
        switch (opts.optimizerType) {
            case OptimizerType::Simple:
                return std::make_unique<SimpleOptimizer>(opts.embeddingDimension, numVertices,
                                                         opts.simpleOptMaxDisplacement);
            case OptimizerType::Adam:
                return std::make_unique<AdamOptimizer>(opts.embeddingDimension, numVertices, 0.9, 0.999, 1e-8);
        }
        return std::make_unique<AdamOptimizer>(opts.embeddingDimension, numVertices, 0.9, 0.999, 1e-8);
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
    double repellingForce(NodeId v, NodeId u, VecBuffer<1>& forceBuffer);
    void applyGravityCentre();

    /**
     * Computes the relative node displacement of the step just applied
     * (mean per-node movement since previousPositions / radius of gyration)
     * and feeds it to the displacement monitor. Must run after the positions
     * have been updated and recentred.
     */
    void observeDisplacement();

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
    // initializeState controls whether the constructor sets a random starting layout and
    // the degree/unit weights. Pass false only when the caller assigns coordinates AND
    // weights immediately afterwards (e.g. LayeredEmbedder layer expansion): skipping the
    // throwaway random init avoids generating and sorting a full layout that is discarded,
    // which is pure wasted work proportional to the layer size on every expansion.
    WembedEmbedder(const Graph& g,
                      const EmbedderOptions &opts,
                      const std::shared_ptr<util::Timer> &timer_ptr = std::make_shared<util::Timer>(),
                      bool initializeState = true)
                      : EmbedderInterface(g, opts),
                        timer(timer_ptr),
                        invExpWeights(g.getNumVertices()),
                        lossPerNode(g.getNumVertices()),
                        previousPositions(opts.embeddingDimension, g.getNumVertices()),
                        perNodeDisplacement(g.getNumVertices()),
                        perNodeRadiusSq(g.getNumVertices()),
                        posOptimizer(makePosOptimizer(opts, g.getNumVertices())),
                        convergenceMonitor(std::make_unique<ConvergenceMonitor>(opts.stopLossTol, opts.stopLossPatience,
                                                                                opts.lossSmoothingFactor,
                                                                                opts.lossRateWindow)),
                        displacementMonitor(std::make_unique<DisplacementMonitor>(opts.stopDisplacementTol,
                                                                                  opts.stopDisplacementPatience)),
                        lrScheduler(makeLRScheduler(opts, *convergenceMonitor))
    {
        if (!initializeState) {
            return;  // caller assigns coordinates and weights right after construction
        }

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
