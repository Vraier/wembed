#pragma once

#include <CLI/CLI.hpp>
#include "EmbedderOptions.hpp"
#include "GraphIO.hpp"

struct Options {
    // Input / Output
    std::string graphPath = "";
    std::string embeddingPath = "";
    bool showTimings = false;

    // Visualization
    std::string svgPath = "";
    bool animate = false;

    // Embedder Options
    bool layeredEmbedding = false;
    EmbedderOptions embedderOptions;
};