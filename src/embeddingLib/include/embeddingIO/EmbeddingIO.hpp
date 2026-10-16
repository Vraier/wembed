#pragma once

#include <memory>
#include <string>
#include <vector>

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
    PoincareEmb = 7,
    InfNormEmb = 8,
    AdditiveEmb = 9
};

class EmbeddingIO {
   public:
    static std::unique_ptr<Embedding> parseEmbedding(EmbeddingType type,
                                                     const std::vector<std::vector<flt_t>>& coordinates, int lpNorm);

    /**
     * Reads coordinates for a graph from a file.
     * The first entry is the NodeId, followed by d coordinates
     *
     * Ignores lines starting with the comment symbol.
     * Assumes single coordinate values are separated by the delimiter.
     *
     * Assumes the ids in the file to be consecutive starting from 0.
     */
    static std::vector<std::vector<flt_t>> readCoordinatesFromFile(std::string filePath, std::string comment = "#",
                                                                    std::string delimiter = ",");

    /**
     * Used to get the weights for weighted embeddings.
     *
     * Splits the last column away and writes it into a new vector
     */
    static std::pair<std::vector<std::vector<flt_t>>, std::vector<flt_t>> splitLastColumn(
        const std::vector<std::vector<flt_t>>& coordinates);

    /**
     * Used to get the weights for mercator embeddings.
     */
    static std::pair<std::vector<flt_t>, std::vector<std::vector<flt_t>>> splitFirstColumn(
        const std::vector<std::vector<flt_t>>& coordinates);

    /**
     * mapping maps NodeIds of the input file to nodeIds in the position vector
     */
    static void writeCoordinates(std::string filePath, const std::vector<std::vector<flt_t>>& positions,
                                 const std::vector<flt_t>& weights);
    static void writeCoordinates(std::string filePath, const std::vector<std::vector<flt_t>>& positions);
};
