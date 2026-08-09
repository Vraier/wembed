#pragma once

#include <string>
#include <CLI/CLI.hpp>
#include "wembed.h"

struct Options {
    int seed = -1;

    // Input / Output
    std::string graphPath;
    std::string embeddingPath;

    std::string inputEmbeddingPath;
    std::string embeddingComment = "%";
    std::string embeddingDelimiter = ",";

    std::string tracePath;

    bool showTimings = false;

    wembed::Options embedderOptions;
};
