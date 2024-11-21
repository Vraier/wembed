#pragma once

#include <CLI/CLI.hpp>
#include "EmbedderOptions.hpp"
#include "GraphIO.hpp"

struct Options {
    std::string graphPath = "";
    std::string embeddingPath = "";

    bool showTimings = false;

    EmbedderOptions embedderOptions;
};