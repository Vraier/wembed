#pragma once

#include <cmath>
#include <map>
#include <string>

enum class OptimizerType { Simple = 0, Adam = 1 };

enum class WeightType { Unit = 0, Degree = 1, Original = 2 };

enum class IndexType { RTree = 0, SNN = 1 };

inline std::map<OptimizerType, std::string> optimizerTypeMap = {{OptimizerType::Simple, "Simple"},
                                                                {OptimizerType::Adam, "Adam"}};

inline std::map<WeightType, std::string> weightTypeMap = {
    {WeightType::Unit, "Unit"}, {WeightType::Degree, "Degree"}, {WeightType::Original, "Original"}};

inline std::map<IndexType, std::string> indexTypeMap = {{IndexType::RTree, "RTree"}, {IndexType::SNN, "SNN"}};

struct EmbedderOptions {
    int embeddingDimension = 4;
    double dimensionHint = -1.0;  // hint for the dimension of the input graph

    // Force parameters
    WeightType weightType = WeightType::Degree;  // determines how the weights are initially set
    int numNegativeSamples = -1;           // determines the number of negative samples. -1 means spacial index is used.
    IndexType indexType = IndexType::SNN;  // determines the type of index used for the embedding
    double IndexSize = 1.0;                // fraction of nodes that get inserted into the spacial index
    double doublingFactor = 2.0;           // determines how the weight buckets are calculated
    double relativePosMinChange = std::pow(10.0, -8);  // used to determine when the embedding can be halted
    double attractionScale = 1.0;                      // factor by which attracting forces are scaled
    double repulsionScale = 1.0;                       // factor by which repulsion forces are scaled
                                                       //(usually best to set to same as attraction)
    double edgeLength = 1.0;
    double expansionStretch = 1.0; // relative amount by which the embeddings is stretched during layer expansion

    // regarding weights
    double weightLearningRate = 0.0;  // learning rate for weights
    double weightPenatly = 0.0;       // how strong too large weights are penalized
    bool dumpWeights = false;         // if set, the weights will be dumped to a file

    // Gradient descent parameters
    OptimizerType optimizerType = OptimizerType::Adam;
    double coolingFactor = 0.99;  // strong influence on runtime but increases quality
    double learningRate = 10;     // learning rate
    int maxIterations = 1000;
    bool useInfNorm = false;  // if set, the infinity norm will be used instead of euclidean norm
};