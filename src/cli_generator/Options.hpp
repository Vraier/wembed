#pragma once

#include <string>

struct Options {
    // files
    std::string girgFile = "";
    std::string girgCoords = "";

    // generation parameters
    int seed = -1;        // -1 uses time as seed
    int numNodes = 1000;  // maximum number of nodes
    double ple = 2.5;     // power law exponent
    double averageDegree = 15;
    int genDimension = 2;
    double temperature = 0.1;
    bool torus = false;  // embed on torus
};
