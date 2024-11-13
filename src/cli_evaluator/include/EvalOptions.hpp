#pragma once

#include <boost/program_options.hpp>

#include "OptionParser.hpp"

struct OptionValues {
    bool help;
    int seed = -1;

    std::string edgeListFile = "";
    std::string logFile = "";
    std::string timeFile = "";
    int logType = 0;

    std::string coordFile = "";
    int embType = -1;
    std::string coordComment = "%";
    std::string coordDelimiter = ",";

    // evaluation
    bool printMetricNames;
    int embDimension = 0;

    // TODO(JP) sort these entries correctly
    // edge histogram
    std::string histFile = "";
    std::string nodeHistFile = "";
    double edgeSampleScale = 10.0;
    int numBuckets = 1000;
    // amount of nodes that get sampled during reconstruction (each node has linear runtime!!)
    double nodeSamplePercent = 1.0;
};

/**
 * Parses the options from the command line into the option_values struct.
 * Can check for validity of the options.
 * Can print information about the usage of the program.
 */
class EvalOptions {
   public:
    OptionValues values;
    OptionParser parser;
    EvalOptions();

   private:
    std::map<int, std::string> embeddingPossibilities();
    std::map<int, std::string> configPossibilities();
};
