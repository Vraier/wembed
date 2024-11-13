#pragma once

#include <boost/program_options.hpp>
#include <iostream>

#include "OptionParser.hpp"
#include "Partitioner.hpp"
#include "EmbedderOptions.hpp"

/**
 * Determines what graph is beeing embedded.
 * This includes if the graph should come from a random model or be read from a file
 */
struct GraphOptions {
    std::string graphPath = "";
    int graphFileType = 0;
    std::string girgWeights = "";
    int graphSize = 20;
};

/**
 * Determines how the graph embedding should be exported
 */
struct WriterOptions {
    std::string svgPath = "";
    std::string ipePath = "";
    std::string coordPath = "";
};

struct OptionValues {
    bool help;
    int seed;
    bool animate = false;
    bool show_timings = false;
    int dimension = 2;

    GraphOptions generatorOptions;
    EmbedderOptions embedderOptions;
    PartitionerOptions partitionerOptions;
    WriterOptions writerOptions;
};

class EmbOptions {
   public:
    OptionValues values;
    OptionParser parser;

    EmbOptions();
    void fillDefaultValues();
    bool checkValidity();

   private:
    std::map<int, std::string> graphFilePossibilities();

    std::map<int, std::string> weightPossibilities();

    std::map<int, std::string> optimizerPossibilities();
    std::map<int, std::string> embedderPossibilities();
    std::map<int, std::string> weightApproximationPossibilities();
    std::map<int, std::string> approxSelectionPossibilities();
    std::map<int, std::string> samplingHeuristicPossibilities();

    std::map<int, std::string> partitionPossibilities();
    std::map<int, std::string> partitionOrderPossibilities();
};
