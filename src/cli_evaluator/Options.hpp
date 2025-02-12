#pragma once

#include "EmbedderOptions.hpp"
#include "GraphIO.hpp"
#include "EmbeddingIO.hpp"
#include "ConfigParser.hpp"

struct Options {
    bool headerOnly = false;

    // input files
    std::string edgeListPath = "";
    std::string edgeListComment = "#";
    std::string edgeListDelimiter = " ";
    
    std::string embeddingPath = "";
    std::string embeddingComment = "%";
    std::string embeddingDelimiter = ",";
    EmbeddingType embType = EmbeddingType::WeightedEmb;
    
    std::string logPath = "";
    LogType logType = LogType::None;

    std::string timePath = "";

    // evaluation parameters
    int seed = -1;
    double edgeSampleScale = 10.0; // how many more non edges get sampled than edges
    double nodeSamplePercent = 0.01; // amount of nodes that get sampled during reconstruction metric (each node has linear runtime!!). Capped at max 5000 nodes
};

std::map<EmbeddingType, std::string> embeddingTypeMap = {
    {EmbeddingType::WeightedEmb, "Weighted"},
    {EmbeddingType::EuclideanEmb, "Euclidean"},
    {EmbeddingType::DotProductEmb, "DotProduct"},
    {EmbeddingType::CosineEmb, "Cosine"},
    {EmbeddingType::MercatorEmb, "Mercator"},
    {EmbeddingType::WeightedNoDimEmb, "WeightedNoDim"},
    {EmbeddingType::WeightedInfEmb, "WeightedInf"},
    {EmbeddingType::PoincareEmb, "Poincare"}
};

std::map<LogType, std::string> logTypeMap = {
    {LogType::None, "none"},
    {LogType::WEmbed, "WEmbed"},
    {LogType::CSV, "general csv config"}
};