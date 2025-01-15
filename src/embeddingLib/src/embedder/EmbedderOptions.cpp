#include "EmbedderOptions.hpp"

std::map<WeightType, std::string> weightTypeMap = {
    {WeightType::Unit, "Unit"},
    {WeightType::Degree, "Degree"},
    {WeightType::Original, "Original"}
};

std::string optimizerTypeToString(OptimizerType type) {
    switch (type) {
        case Simple:
            return "Simple";
        case Adam:
            return "Adam";
        default:
            return "Unknown";
    }
}

std::string embedderTypeToString(EmbedderType type) {
    switch (type) {
        case hFruchtermann:
            return "hFruchtermann";
        case hSigmoid:
            return "hSigmoid";
        case hFruchtermannSigmoid:
            return "hFruchtermannSigmoid";
        case hLinear:
            return "hLinear";
        case hFruchtermannLinear:
            return "hFruchtermannLinear";
        case sFruchtermann:
            return "sFruchtermann";
        case sLocal:
            return "sLocal";
        case sSigmoid:
            return "sSigmoid";
        case slowFruchtermann:
            return "slowFruchtermann";
        case slowF2V:
            return "slowF2V";
        case slowSigmoidEuclidean:
            return "slowSigmoidEuclidean";
        case slowMaxent:
            return "slowMaxent";
        case slowF2VNormed:
            return "slowF2VNormed";
        case slowSigmoidWeighted:
            return "slowSigmoidWeighted";
        case slowSigmoidNoDim:
            return "slowSigmoidNoDim";
        case slowLinear:
            return "slowLinear";
        case slowSampling:
            return "slowSampling";
        default:
            return "Unknown";
    }
}

std::string weightTypeToString(WeightType type) {
    switch (type) {
        case Unit:
            return "Unit";
        case Degree:
            return "Degree";
        case Original:
            return "Original";
        default:
            return "Unknown";
    }
}

std::string samplingTypeToString(SamplingHeuristicType type) {
    switch (type) {
        case SamplingHeuristicType::Quadratic:
            return "Quadratic";
        case SamplingHeuristicType::Random:
            return "Random";
        case SamplingHeuristicType::Girg:
            return "Girg";
        case SamplingHeuristicType::BFS:
            return "BFS";
        case SamplingHeuristicType::Distance:
            return "Distance";
        case SamplingHeuristicType::RTree:
            return "RTree";
        default:
            return "Unknown";
    }
}