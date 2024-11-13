#pragma once

#include "OptionParser.hpp"

struct OptionValues {
    bool help;
    int seed = -1;

    // graph generation
    std::string girgFile = "";
    std::string girgCoords = ""; // currently not used

    // generation parameters
    int numNodes = 1000;
    double ple = 20;  // power law exponent
    double averageDegree = 15;
    int genDimension = 2;
    double temperature = 0.0;
    bool torus = false;  // embed on torus
};

class GenOptions {
   public:
    OptionValues values;
    OptionParser parser;
    GenOptions();

    bool checkValidity();
};