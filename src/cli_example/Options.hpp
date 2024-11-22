#pragma once

#include <CLI/CLI.hpp>
#include "EmbedderOptions.hpp"
#include "GraphIO.hpp"

struct Options {
    std::string graphPath = "";
    std::string embeddingPath = "";

    bool showTimings = false;
    #ifdef EMBEDDING_USE_SFML
    bool animate = false;
    #endif

    EmbedderOptions embedderOptions;
};