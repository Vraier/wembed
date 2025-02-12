#pragma once

#include <cmath>
#include <string>
#include <map>

enum OptimizerType { Simple = 0, Adam = 1 };

enum EmbedderType {
    // hierarchical embedders
    hFruchtermann = 0,
    hSigmoid = 1,
    hFruchtermannSigmoid = 2,
    hLinear = 3,
    hFruchtermannLinear = 4,

    // slow embedders
    sFruchtermann = 7,
    sLocal = 8,
    sSigmoid = 9,

    // (new) generation of slow embedders
    slowFruchtermann = 10,
    slowF2V = 11,
    slowSigmoidEuclidean = 12,
    slowMaxent = 13,
    slowF2VNormed = 14,
    slowSigmoidWeighted = 15,
    slowSigmoidNoDim = 16,
    slowLinear = 17,
    slowSampling = 18,
};

enum WeightType { Unit = 0, Degree = 1, Original = 2 };

extern std::map<WeightType, std::string> weightTypeMap;

enum class SamplingHeuristicType { Quadratic = 0, Random = 1, Girg = 2, BFS = 3, Distance = 4, RTree = 5, DiscANN = 6 };

// TODO: clean these up and remove unused ones
struct EmbedderOptions {
    // Optimization
    OptimizerType optimizerType = Adam;

    EmbedderType embedderType = slowSampling;
    bool useOriginalCoords =
        false;  // if set, the original coordinates are used as initial positions (if they are provided in the input)
    bool useInfNorm = false;   // if set, the infinity norm will be used instead of euclidean norm
    double dimensionHint = 4.0;  // hint for the dimension of the input graph
    int embeddingDimension = 4;
    double relativePosMinChange = std::pow(10.0, -8);  // used to determine when the embedding can be halted

    // Gradient descent parameters
    double coolingFactor = 0.99; // strong influence for runtime
    double speed = 10;
    int maxIterations = 2000;

    // approximation
    int maxApproxComparisons = 50;
    int approxSelectionType = 0;         // what method is used to traverse the tree
    SamplingHeuristicType samplingType =
        SamplingHeuristicType::RTree;  // how does a subset of negative samples for repelling forces get selected
    int numNegativeSamples = -1;       // determins the number of negative samples.
    bool uniformSampling = false;      // determines if every nodes gets the same amount of samples
    double doublingFactor = 2.0;       // determines how the weight buckets are calculated

    // Regarding weights
    WeightType weightType = Degree;                       // determines how the weights are initially set
    bool staticWeights = true;                            // keep original weights and don't change them later
    int weightApproximation = 0;                          // determines how weights are updated
    double relativeWeightMinChange = std::pow(10.0, -3);  // determines when the weight updates are stopped
    double weightSpeed = 0.01;                            // determines how fast the weights are updated
    double weightCooling = 0.987;                         // determines how fast the weight speed slows down
    int numWeightSamplesPerCluster = 10;

    // Fruchterman
    double cRep = 2.0;
    double cSpring = 0.5;
    double maxDisplacement = 10;
    double forceExponent = 2;
    bool relaxedEdgeLength = false;  // determines if the edges should be exactly the ideal edge length or only smaller
    bool neighborRepulsion = false;  // determines if repulsion forces are also calculated between neighboring nodes

    // local embedder
    double sigmoidScale = 1.0;
    double sigmoidLength = 1.0;

    // MaxEnt
    double maxentInitialAlpha = 0.1;
    double maxentAlphaCooling = 0.3;
    double maxentMinAlpha = 0.008;
    int rounds = 10;
    int iterations = 50;
};