#pragma once

#include <CLI/CLI.hpp>
#include "EmbedderOptions.hpp"
#include "GraphIO.hpp"

struct Options {
    int seed = -1;

    // Input / Output
    std::string graphPath = "";
    bool bipartite = false;

    std::string embeddingPath = "";

    std::string inputEmbeddingPath = "";
    std::string embeddingComment = "%";
    std::string embeddingDelimiter = ",";

    bool showTimings = false;

    // Visualization
    bool animate = false;

    // Embedder Options
    bool layeredEmbedding = false;
    EmbedderOptions embedderOptions;
};