#pragma once

#include <map>
#include <string>

enum class OptimizerType { Simple = 0, Adam = 1 };

enum class WeightType { Unit = 0, Degree = 1 };

enum class IndexType {SNN = 1, Sprk = 2 };

enum class LRScheduleType { ExponentialCooling = 0, LossAdaptive = 1 };

enum class StopCriterionType { Displacement = 0, Loss = 1 };

inline std::map<OptimizerType, std::string> optimizerTypeMap = {{OptimizerType::Simple, "Simple"},
                                                                {OptimizerType::Adam, "Adam"}};

inline std::map<LRScheduleType, std::string> lrScheduleTypeMap = {
    {LRScheduleType::ExponentialCooling, "ExponentialCooling"}, {LRScheduleType::LossAdaptive, "LossAdaptive"}};

inline std::map<StopCriterionType, std::string> stopCriterionTypeMap = {
    {StopCriterionType::Displacement, "Displacement"},
    {StopCriterionType::Loss, "Loss"}};

inline std::map<WeightType, std::string> weightTypeMap = {
    {WeightType::Unit, "Unit"}, {WeightType::Degree, "Degree"}};

inline std::map<IndexType, std::string> indexTypeMap = { {IndexType::SNN, "SNN"}, {IndexType::Sprk, "Sprk"}};

struct EmbedderOptions {
    int embeddingDimension = 4;
    double dimensionHint = -1.0;  // hint for the dimension of the input graph

    // Force parameters
    WeightType weightType = WeightType::Degree;  // determines how the weights are initially set
    int numNegativeSamples = -1;           // determines the number of negative samples. -1 means spacial index is used.
    IndexType indexType = IndexType::Sprk;  // determines the type of index used for the embedding
    double IndexSize = 1.0;                // fraction of nodes that get inserted into the spacial index
    double doublingFactor = 2.0;           // determines how the weight buckets are calculated
    double attractionScale = 1.0;                   // factor by which attracting forces are scaled
    double repulsionScale = 1.0;                    // factor by which repulsion forces are scaled
                                                    //(usually best to set to same as attraction)
    double centreScale = 0.0; //factor by which each node is drawn to the centre
    double edgeLength = 1.0;
    double expansionStretch = 1.0;  // relative amount by which the embeddings is stretched during layer expansion

    bool additiveWeights = false;

    // Gradient descent parameters
    OptimizerType optimizerType = OptimizerType::Adam;
    int maxIterations = 10000;
    double simpleOptMaxDisplacement = 1.0;  // per-step displacement cap (SimpleOptimizer only)

    // Learning rate schedule parameters (lr* prefix). Every parameter states which schedules read it.
    LRScheduleType lrScheduleType = LRScheduleType::ExponentialCooling;
    double learningRate = 10;       // initial learning rate (both schedules)
    int warmupSteps = 20;           // linear LR ramp-up over the first steps (both schedules)
    double lrCoolingFactor = 0.995;  // per-step multiplicative decay (ExponentialCooling only);
                                    // strong influence on runtime but increases quality
    double lrDecayFactor = 0.5;     // multiplicative drop on a decay event (LossAdaptive only)
    double lrDecayThreshold = 1e-2;  // decay when the loss-decrease rate stays below this (LossAdaptive only)
    int lrAdaptPatience = 20;       // consecutive in-zone steps before a decay OR growth event (LossAdaptive only)
    double lrGrowthFactor = 1.0;    // multiplicative growth while the loss keeps decreasing fast
                                    // (LossAdaptive only; 1.0 disables growth -> pure plateau decay)
    double lrGrowthThreshold = 1e-1;  // grow when the loss-decrease rate stays above this (LossAdaptive only)

    // Stopping criterion (maxIterations always applies as a hard cap).
    StopCriterionType stopCriterion = StopCriterionType::Loss;  // which signal terminates the run

    // Displacement stopping criterion (StopCriterionType::Displacement).
    double stopDisplacementTol = 3e-4;  // relative per-step node movement (mean displacement / radius of
                                        // gyration) below which the layout counts as settled
    int stopDisplacementPatience = 5;   // settled steps in a row before stopping

    // Shared loss-progress signal (loss* prefix): windowed relative loss decrease rate(t),
    // consumed by both the loss stop criterion and the LossAdaptive schedule.
    double lossSmoothingFactor = 0.3;  // EMA weight of the newest loss sample before the monitor sees it
                                       // (1.0 disables smoothing); a light denoise on rate(t)
    int lossRateWindow = 30;           // steps over which the relative loss-decrease rate is measured
                                       // (a real window; per-step change is too noisy to threshold)

    // Loss stagnation stopping criterion (StopCriterionType::Loss).
    double stopLossTol = 1e-3;   // ftol: converged once rate(t) stays below this (relative decrease over the window)
    int stopLossPatience = 50;   // sub-tolerance steps in a row before stopping.
                                 // Recommended ordering: lrGrowthThreshold > lrDecayThreshold >= stopLossTol,
                                 // and lrAdaptPatience < stopLossPatience so LossAdaptive cools a few times first.
};
