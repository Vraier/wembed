#include "EmbOptions.hpp"

#include "EmbedderFactory.hpp"

namespace po = boost::program_options;

EmbOptions::EmbOptions() : parser("Embedder") {
    GraphOptions genDefault;
    EmbedderOptions embDefault;
    PartitionerOptions ptDefault;
    WriterOptions writeDefault;

    std::string GraphFileDescription =
        "File format of the graph file, " + parser.getDescriptionOfEnumOption(graphFilePossibilities());
    std::string OptimizerDesc =
        "Algorithm that is used to optimize the embedding, " + parser.getDescriptionOfEnumOption(optimizerPossibilities());
    std::string EmbedderDesc =
        "Algorithm that is used to embed the graph, " + parser.getDescriptionOfEnumOption(embedderPossibilities());
    std::string InitialWeightDesc =
        "Determines how the initial weights are set, " + parser.getDescriptionOfEnumOption(weightPossibilities());
    std::string WeightApproximationDesc =
        "Determines how the weights are updated, " + parser.getDescriptionOfEnumOption(weightApproximationPossibilities());
    std::string PartitionerDesc =
        "Algorithm that is used to partition the graph, " + parser.getDescriptionOfEnumOption(partitionPossibilities());
    std::string PartitionOrderDesc =
        "Orders for the label propagation, " + parser.getDescriptionOfEnumOption(partitionOrderPossibilities());
    std::string ApproxSelectionDesc =
        "Determines how the tree for rep forces is traversed, " + parser.getDescriptionOfEnumOption(approxSelectionPossibilities());
    std::string SamplingHeuristicDesc =
        "Determines how the negative samples are choosen, " + parser.getDescriptionOfEnumOption(samplingHeuristicPossibilities());

    parser.description.add_options()
        // general settings
        ("seed,s", po::value<int>(&values.seed)->default_value(-1),
         "Seed for random number generation. -1 Means using time for seed.")  //
        ("help,h", po::bool_switch(&values.help)->default_value(false),
         "Prints an overview of all options.")  //
        ("animate,a", po::bool_switch(&values.animate)->default_value(false),
         "If the flag is set, an animation of the embedding will be played. Looks nice and is good for debugging.")  //
        ("show-timings,t", po::bool_switch(&values.show_timings)->default_value(false),
         "If the flag is set, a breakdown of the embedding timings is printed at the end.")  //
        ("dimension,d", po::value<int>(&values.dimension)->default_value(2),
         "The dimension in which the graph is embedded in.")  //
        // graph generation settings
        ("gg-file-type", po::value<int>(&values.generatorOptions.graphFileType)->default_value(genDefault.graphFileType),
         GraphFileDescription.c_str())  //
        ("gg-file", po::value<std::string>(&values.generatorOptions.graphPath)->default_value(genDefault.graphPath),
         "Filepath to the graph that will be embedded.")  //
        ("gg-nodes,n", po::value<int>(&values.generatorOptions.graphSize)->default_value(genDefault.graphSize),
         "number of vertices of the generated graph")  //
        ("gg-girg-weights", po::value<std::string>(&values.generatorOptions.girgWeights)->default_value(genDefault.girgWeights),
         "File that contains information about weights and coordinates of the used graph")  //
        // partitioner options
        ("pt-type", po::value<int>(&values.partitionerOptions.partitionType)->default_value(ptDefault.partitionType),
         PartitionerDesc.c_str())  //
        ("pt-iterations", po::value<int>(&values.partitionerOptions.maxIterations)->default_value(ptDefault.maxIterations),
         "Maximum number of iterations for the label propagation")  //
        ("pt-cluster", po::value<int>(&values.partitionerOptions.maxClusterSize)->default_value(ptDefault.maxClusterSize),
         "Maximum cluster Size for the clusters in a partition. Can be ignored if hierarchy would become too large.")  //
        ("pt-coarsest", po::value<int>(&values.partitionerOptions.finalGraphSize)->default_value(ptDefault.finalGraphSize),
         "Size of the coarsest graph in partition hierarchy.")  //
        ("pt-order", po::value<int>(&values.partitionerOptions.orderType)->default_value(ptDefault.orderType),
         PartitionOrderDesc.c_str())  //
         ("pt-hierarchies", po::value<int>(&values.partitionerOptions.numHierarchies)->default_value(ptDefault.numHierarchies), 
         "Number of hierarchies that are used to approximate forces.")  //
        // Embedder options
        ("emb-optimizer", po::value<int>(&values.embedderOptions.optimizerType)->default_value(embDefault.optimizerType),
         OptimizerDesc.c_str()) //
        ("emb-type", po::value<int>(&values.embedderOptions.embedderType)->default_value(embDefault.embedderType),
         EmbedderDesc.c_str())  //
        ("emb-crep", po::value<double>(&values.embedderOptions.cRep)->default_value(embDefault.cRep),
         "repulsion constant used during force directed calculation")  //
        ("emb-use-coords", po::bool_switch(&values.embedderOptions.useOriginalCoords)->default_value(embDefault.useOriginalCoords),
         "Uses the coordinates of the ground truth for the initial embedding.")  //
        ("emb-cspring", po::value<double>(&values.embedderOptions.cSpring)->default_value(embDefault.cSpring),
         "spring constant used during force directed calculation")  //
        ("emb-cooling-factor", po::value<double>(&values.embedderOptions.coolingFactor)->default_value(embDefault.coolingFactor),
         "cooling speed of the force directed calculation. Must be between 1 and 0.")  //
        ("emb-iterations", po::value<int>(&values.embedderOptions.maxIterations)->default_value(embDefault.maxIterations),
         "maximum number of iterations for the force directed calculation")  //
        ("emb-approx-factor", po::value<int>(&values.embedderOptions.maxApproxComparisons)->default_value(embDefault.maxApproxComparisons),
         "The maximum coarsening used during force approximation")  //
        ("emb-use-inf-norm", po::bool_switch(&values.embedderOptions.useInfNorm)->default_value(embDefault.useInfNorm),
         "If set, the infinity norm is used for the force calculation")  //
        ("emb-approx-selection", po::value<int>(&values.embedderOptions.approxSelectionType)->default_value(embDefault.approxSelectionType),
         ApproxSelectionDesc.c_str())  //
        ("emb-output-sampling", po::bool_switch(&values.embedderOptions.outputSamplingMetrics)->default_value(embDefault.outputSamplingMetrics),
         "If set, metrics about sampling will be calculated (results in quadratic runtime).")  //
        ("emb-sampling-heuristic", po::value<int>(&values.embedderOptions.samplingHeuristic)->default_value(embDefault.samplingHeuristic),
         SamplingHeuristicDesc.c_str())  //
        ("emb-neg-samples", po::value<int>(&values.embedderOptions.numNegativeSamples)->default_value(embDefault.numNegativeSamples),
         "Determines the number of negative samples.")  //
        ("emb-uni-sample", po::bool_switch(&values.embedderOptions.uniformSampling)->default_value(embDefault.uniformSampling),
         "Determines if every nodes gets the same amount of samples")  //
        ("emb-doubling-factor", po::value<double>(&values.embedderOptions.doublingFactor)->default_value(embDefault.doublingFactor),
         "Determines how the weight buckets are calculated")  //
        ("emb-dim-hint", po::value<double>(&values.embedderOptions.dimensionHint)->default_value(embDefault.dimensionHint),
         "Hint for the dimension of the input graph. If set to -1 the same dimension as the embedding dimension is used.")  //
        ("emb-force-exp", po::value<double>(&values.embedderOptions.forceExponent)->default_value(embDefault.forceExponent),
         "Exponent for the force calculation. A larger exponent means more aggressive forces. For -1 the embedding dimension is used.")  //
        ("emb-relax-edges", po::bool_switch(&values.embedderOptions.relaxedEdgeLength)->default_value(embDefault.relaxedEdgeLength),
         "If set, the edges can be smaller that the ideal edge length and do not have to be the exact size.")  //
        ("emb-neighbor-repulsion", po::bool_switch(&values.embedderOptions.neighborRepulsion)->default_value(embDefault.neighborRepulsion),
         "If set, repulsion forces are also calculated between neighboring nodes.")  //
        ("emb-speed", po::value<double>(&values.embedderOptions.speed)->default_value(embDefault.speed),
         "Speed of the force directed simulation. Lower speed is more stable but converges slower")  //
        ("emb-max-displacement", po::value<double>(&values.embedderOptions.maxDisplacement)->default_value(embDefault.maxDisplacement),
         "maximum distance a node is allowed to be moved in  any dimension in a single iteration")  //
        ("emb-min-change", po::value<double>(&values.embedderOptions.relativePosMinChange)->default_value(embDefault.relativePosMinChange),
         "Determines when the position steps stop")  //
        ("emb-sigmoid-scale", po::value<double>(&values.embedderOptions.sigmoidScale)->default_value(embDefault.sigmoidScale),
         "Determines how aggressive the sigmoid functions moves nodes")  //
        ("emb-sigmoid-length", po::value<double>(&values.embedderOptions.sigmoidLength)->default_value(embDefault.sigmoidLength),
         "Determines how long the edges for sigmoid forces are")  //
        // regarding weights
        ("emb-w-init", po::value<int>(&values.embedderOptions.weightType)->default_value(embDefault.weightType),
         InitialWeightDesc.c_str())  //
        ("emb-w-static", po::bool_switch(&values.embedderOptions.staticWeights)->default_value(embDefault.staticWeights),
         "Keeps the original weights and does not change them during calculation.")  //
        ("emb-w-approx", po::value<int>(&values.embedderOptions.weightApproximation)->default_value(embDefault.weightApproximation),
         WeightApproximationDesc.c_str())  //
        ("emb-w-min-change", po::value<double>(&values.embedderOptions.relativeWeightMinChange)->default_value(embDefault.relativeWeightMinChange),
         "Determines when the weight steps stop")  //
        ("emb-w-speed", po::value<double>(&values.embedderOptions.weightSpeed)->default_value(embDefault.weightSpeed),
         "The speed with witch weights are updated")  //
        ("emb-w-samples", po::value<int>(&values.embedderOptions.numWeightSamplesPerCluster)->default_value(embDefault.numWeightSamplesPerCluster),
         "Determines how many weights samples are choosen to approximate good weights")  //
        // Writer Options
        ("wrt-svg", po::value<std::string>(&values.writerOptions.svgPath)->default_value(writeDefault.svgPath),
         "Filepath to the svg that represents a graph drawing")  //
        ("wrt-ipe", po::value<std::string>(&values.writerOptions.ipePath)->default_value(writeDefault.ipePath),
         "Filepath to the ipe that represents a graph drawing")  //
        ("wrt-coord", po::value<std::string>(&values.writerOptions.coordPath)->default_value(writeDefault.coordPath),
         "Filepath to a file containing coordinates for every node");  //
}

void EmbOptions::fillDefaultValues() {
    if (values.embedderOptions.forceExponent < 0) {
        values.embedderOptions.forceExponent = values.dimension;
    }
    if (values.embedderOptions.dimensionHint < 0) {
        values.embedderOptions.dimensionHint = values.dimension;
    }
}

bool EmbOptions::checkValidity() {
    if (values.help) return false;
    return true;
}

std::map<int, std::string> EmbOptions::optimizerPossibilities() {
    std::map<int, std::string> res;
    res[0] = "GradientDescent";
    res[1] = "Adam";
    return res;
}

std::map<int, std::string> EmbOptions::embedderPossibilities() {
    std::map<int, std::string> res;
    res[0] = embedderTypeToString(EmbedderType(0));
    res[1] = embedderTypeToString(EmbedderType(1));
    res[2] = embedderTypeToString(EmbedderType(2));
    res[3] = embedderTypeToString(EmbedderType(3));
    res[4] = embedderTypeToString(EmbedderType(4));
    res[7] = embedderTypeToString(EmbedderType(7));
    res[8] = embedderTypeToString(EmbedderType(8));
    res[9] = embedderTypeToString(EmbedderType(9));
    res[10] = embedderTypeToString(EmbedderType(10));
    res[11] = embedderTypeToString(EmbedderType(11));
    res[12] = embedderTypeToString(EmbedderType(12));
    res[13] = embedderTypeToString(EmbedderType(13));
    res[14] = embedderTypeToString(EmbedderType(14));
    res[15] = embedderTypeToString(EmbedderType(15));
    res[16] = embedderTypeToString(EmbedderType(16));
    res[17] = embedderTypeToString(EmbedderType(17));
    res[18] = embedderTypeToString(EmbedderType(18));

    return res;
}

std::map<int, std::string> EmbOptions::graphFilePossibilities() {
    std::map<int, std::string> res;
    res[0] = "EdgList";
    res[1] = "AdjList";
    res[2] = "DotFile";
    return res;
}

std::map<int, std::string> EmbOptions::weightPossibilities() {
    std::map<int, std::string> res;
    res[0] = "Unit";
    res[1] = "Degree";
    res[2] = "Original";
    return res;
}

std::map<int, std::string> EmbOptions::weightApproximationPossibilities() {
    std::map<int, std::string> res;
    res[0] = "LowerBound";
    res[1] = "weightedError";
    res[2] = "exactError";
    res[2] = "sampled";
    return res;
}

std::map<int, std::string> EmbOptions::partitionPossibilities() {
    std::map<int, std::string> res;
    res[0] = "Label";
    res[1] = "Louvain";
    res[2] = "KaHip";
    res[3] = "KaHyPar";
    return res;
}

std::map<int, std::string> EmbOptions::partitionOrderPossibilities() {
    std::map<int, std::string> res;
    res[0] = "Degree";
    res[1] = "Random";
    return res;
}

std::map<int, std::string> EmbOptions::approxSelectionPossibilities() {
    std::map<int, std::string> res;
    res[0] = "Coarsest";
    res[1] = "Priority";
    return res;
}

std::map<int, std::string> EmbOptions::samplingHeuristicPossibilities() {
    std::map<int, std::string> res;
    res[0] = "Quadratic";
    res[1] = "Random";
    res[2] = "Girg";
    res[3] = "BFS";
    res[4] = "Distance";
    res[5] = "RTree";
    return res;
}
