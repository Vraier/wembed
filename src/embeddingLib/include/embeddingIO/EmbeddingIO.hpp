#pragma once

#include <string>
#include <vector>
#include <memory>

#include "Embedding.hpp"
#include "GraphIO.hpp"
#include "VecList.hpp"

enum EmbeddingType {
    WeightedEmb = 0,
    EuclideanEmb = 1,
    DotProductEmb = 2,
    CosineEmb = 3,
    MercatorEmb = 4,
    WeightedNoDimEmb = 5,
    WeightedInfEmb = 6,
    PoincareEmb = 7
};

class EmbeddingIO {
   public:
    static std::unique_ptr<Embedding> parseEmbedding(EmbeddingType type,
                                                     const std::vector<std::vector<double>>& coordinates);

    /**
     * Reads coordinates for a graph from a file.
     * The first entry is the NodeId, followed by d coordinates
     *
     * Ignores lines starting with the comment symbol.
     * Assumes single coordinate values are separated by the delimiter.
     *
     * Assumes the ids in the file to be consecutive starting from 0.
     */
    static std::vector<std::vector<double>> readCoordinatesFromFile(std::string filePath, std::string comment = "#",
                                                                    std::string delimiter = ",");

    /**
     * Used to get the weights for weighted embeddings.
     *
     * Splits the last column away and writes it into a new vector
     */
    static std::pair<std::vector<std::vector<double>>, std::vector<double>> splitLastColumn(
        const std::vector<std::vector<double>>& coordinates);

    /**
     * Used to get the weights for mercator embeddings.
     */
    static std::pair<std::vector<double>, std::vector<std::vector<double>>> splitFirstColumn(
        const std::vector<std::vector<double>>& coordinates);

    /**
     * mapping maps NodeIds of the input file to nodeIds in the position vector
     */
    static void writeCoordinates(std::string filePath, const std::vector<std::vector<double>>& positions,
                                 const std::vector<double>& weights);
    static void writeCoordinates(std::string filePath, const std::vector<std::vector<double>>& positions);
};
