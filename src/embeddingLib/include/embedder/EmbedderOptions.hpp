#pragma once

#include <map>
#include <string>

enum class OptimizerType { Simple = 0, Adam = 1 };

enum class WeightType { Unit = 0, Degree = 1 };

enum class IndexType {SNN = 1, Sprk = 2 };

enum class LRScheduleType { ExponentialCooling = 0, LossAdaptive = 1 };

inline std::map<OptimizerType, std::string> optimizerTypeMap = {{OptimizerType::Simple, "Simple"},
                                                                {OptimizerType::Adam, "Adam"}};

inline std::map<LRScheduleType, std::string> lrScheduleTypeMap = {
    {LRScheduleType::ExponentialCooling, "ExponentialCooling"}, {LRScheduleType::LossAdaptive, "LossAdaptive"}};

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
    int maxIterations = 1000;
    double simpleOptMaxDisplacement = 1.0;  // per-step displacement cap (SimpleOptimizer only)

    // Learning rate schedule parameters.
    // Every parameter states which schedules read it.
    LRScheduleType lrScheduleType = LRScheduleType::ExponentialCooling;
    double learningRate = 10;     // initial learning rate (both schedules)
    int warmupSteps = 20;         // linear LR ramp-up over the first steps (both schedules)
    double coolingFactor = 0.99;  // per-step multiplicative decay (ExponentialCooling only);
                                  // strong influence on runtime but increases quality
    double decayFactor = 0.5;     // multiplicative drop on a decay event (LossAdaptive only)
    int plateauPatience = 20;     // stagnant steps in a row before a decay event (LossAdaptive only)
    double growthFactor = 1.05;   // multiplicative growth after a significant new best loss
                                  // (LossAdaptive only; 1.0 disables growth)
    double growthRelTol = 3e-3;   // relative loss improvement over the best-so-far that triggers
                                  // an LR increase (LossAdaptive only)

    // Loss stagnation stopping criterion (maxIterations always applies as a hard cap).
    double stopRelTol = 3e-2;  // relative loss improvement below which a step counts as stagnant
                               // (also feeds the stagnation counter that times LossAdaptive's decay events)
    int stopPatience = 50;     // stagnant steps in a row before stopping.
                               // When combined with LossAdaptive keep this >= 3 * plateauPatience
                               // so the schedule gets a few decay events before the embedding stops.
};
