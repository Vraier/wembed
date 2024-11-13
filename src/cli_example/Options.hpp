#pragma once

#include <CLI/CLI.hpp>
#include "EmbedderOptions.hpp"
#include "GraphIO.hpp"

struct Options {
    GraphFileType graphFileType = GraphFileType::edgeList;
    std::string graphPath = "";
    std::string embeddingPath = "";

    bool draw = false;
    bool showTimings = false;

    EmbedderOptions embedderOptions;
};