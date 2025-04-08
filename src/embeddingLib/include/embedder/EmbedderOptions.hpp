#pragma once

#include <cmath>
#include <map>
#include <string>

enum OptimizerType { Simple = 0, Adam = 1 };

enum WeightType { Unit = 0, Degree = 1, Original = 2 };

inline std::map<WeightType, std::string> weightTypeMap = {{Unit, "Unit"}, {Degree, "Degree"}, {Original, "Original"}};

struct EmbedderOptions {
    int embeddingDimension = 4;
    double dimensionHint = -1.0;  // hint for the dimension of the input graph

    // Force parameters
    WeightType weightType = Degree;                    // determines how the weights are initially set
    int numNegativeSamples = -1;                       // determines the number of negative samples.
    double doublingFactor = 2.0;                       // determines how the weight buckets are calculated
    double relativePosMinChange = std::pow(10.0, -8);  // used to determine when the embedding can be halted
    double attractionScale = 1.0;
    double repulsionScale = 0.1;
    double edgeLength = 1.0;

    // Gradient descent parameters
    OptimizerType optimizerType = Adam;
    double coolingFactor = 0.99;  // strong influence for runtime
    double speed = 10;
    int maxIterations = 2000;
    bool useInfNorm = false;  // if set, the infinity norm will be used instead of euclidean norm
};