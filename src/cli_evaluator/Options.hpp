#pragma once

#include <CLI/CLI.hpp>
#include "EmbedderOptions.hpp"
#include "GraphIO.hpp"
#include "EmbeddingIO.hpp"
#include "ConfigParser.hpp"

struct Options {

    // input files
    std::string edgeListPath = "";
    std::string edgeListComment = "%";
    std::string edgeListDelimiter = " ";
    
    std::string embeddingPath = "";
    std::string embeddingComment = "%";
    std::string embeddingDelimiter = ",";
    EmbeddingType embType = EmbeddingType::WeightedEmb;
    
    std::string logPath = "";
    LogType logType = LogType::None;

    std::string timePath = "";



    // output files
    std::string histPath = "";
    std::string nodeHistPath = "";

    // evaluation parameters
    int seed = -1;
    bool printMetricNames = false;
    double edgeSampleScale = 10.0; // how many more non edges get sampled than edges
    double nodeSamplePercent = 0.01; // amount of nodes that get sampled during reconstruction metric (each node has linear runtime!!)
};