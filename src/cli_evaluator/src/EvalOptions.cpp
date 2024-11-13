#include "EvalOptions.hpp"

namespace po = boost::program_options;

EvalOptions::EvalOptions() : parser("Evaluator") {
    OptionValues defaultOpt;

    std::string EmbedderDesc = "Type of the embedding, " + parser.getDescriptionOfEnumOption(embeddingPossibilities());
    std::string LogTypeDec = "Type of log file, " + parser.getDescriptionOfEnumOption(configPossibilities());

    parser.description.add_options()
        // general settings
        ("seed,s", po::value<int>(&values.seed)->default_value(defaultOpt.seed),
         "Seed for random number generation. -1 Means using time for seed.")  //
        ("help,h", po::bool_switch(&values.help)->default_value(false),
         "Prints an overview of all options.")  //
        // graph input settings
        ("edge-list,e", po::value<std::string>(&values.edgeListFile)->default_value(defaultOpt.edgeListFile),
         "Filepath to the edge list of the embedded graph.")  //
        ("coords,c", po::value<std::string>(&values.coordFile)->default_value(defaultOpt.coordFile),
         "Filepath to a csv file containing the coordinates of the nodes.")  //
        ("emb-type", po::value<int>(&values.embType)->default_value(defaultOpt.embType),
         EmbedderDesc.c_str())  //
        ("emb-comment", po::value<std::string>(&values.coordComment)->default_value(defaultOpt.coordComment),
         "The prefix to indicate a comment in the embedding file. Lines with this prefix are ignored.")  //
        ("emb-delimiter", po::value<std::string>(&values.coordDelimiter)->default_value(defaultOpt.coordDelimiter),
         "The string that functions as a delimiter for the coordinates of the embedding file.")  //
        ("log-emb", po::value<std::string>(&values.logFile)->default_value(defaultOpt.logFile),
         "Filepath to the logfile of the generator.")  //
        ("log-type", po::value<int>(&values.logType)->default_value(defaultOpt.logType), LogTypeDec.c_str())
        ("time-file", po::value<std::string>(&values.timeFile)->default_value(defaultOpt.timeFile),
         "Filepath to the file containing the time it took to generate the embedding.")  //
        // general metric information
        ("print-metric-names,p", po::bool_switch(&values.printMetricNames)->default_value(false),
         "Prints the name of each metric.")  //
        // histogram options
        ("edge-samples", po::value<double>(&values.edgeSampleScale)->default_value(defaultOpt.edgeSampleScale),
         "Determines how many more non edges are sampled than edges.")  //
        ("num-buckets", po::value<int>(&values.numBuckets)->default_value(defaultOpt.numBuckets),
         "The sampling rate at which the density of edges is determined.")  //
        ("hist-file", po::value<std::string>(&values.histFile)->default_value(defaultOpt.histFile),
         "File containing histogram information for the edges of the graph")  //
        ("node-hist-file", po::value<std::string>(&values.nodeHistFile)->default_value(defaultOpt.nodeHistFile),
         "File containing histogram information for the nodes of the graph");  //
}

std::map<int, std::string> EvalOptions::embeddingPossibilities() {
    std::map<int, std::string> res;
    res[0] = "weighted";
    res[1] = "euclidean";
    res[2] = "dotProduct";
    res[3] = "cosine";
    res[4] = "mercator";
    res[5] = "weightedNoDim";
    res[6] = "weightedInf";
    res[7] = "poincare";
    return res;
}

std::map<int, std::string> EvalOptions::configPossibilities() {
    std::map<int, std::string> res;
    res[0] = "weightedEmbedder";
    res[1] = "node2Vec";
    res[2] = "csv";
    return res;
}
