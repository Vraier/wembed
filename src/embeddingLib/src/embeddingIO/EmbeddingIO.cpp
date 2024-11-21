#include "EmbeddingIO.hpp"

#include <fstream>
#include <iomanip>

#include "Cosine.hpp"
#include "DotProduct.hpp"
#include "Euclidean.hpp"
#include "GraphIO.hpp"
#include "MercatorEmbedding.hpp"
#include "Poincare.hpp"
#include "StringManipulation.hpp"
#include "WeightedGeometric.hpp"
#include "WeightedGeometricInf.hpp"
#include "WeightedNoDim.hpp"

std::unique_ptr<Embedding> EmbeddingIO::parseEmbedding(EmbeddingType type, const std::vector<std::vector<double>>& coordinates) {
    switch (type) {
        case WeightedEmb:
            // weighted
            {
                LOG_INFO("Constructing weighted geometric embedding");
                auto pair = splitLastColumn(coordinates);
                return std::make_unique<WeightedGeometric>(pair.first, pair.second);
            }

        case EuclideanEmb:
            // euclidean
            {
                LOG_INFO("Constructing euclidean embedding");
                return std::make_unique<Euclidean>(coordinates);
            }
        case DotProductEmb:
            // dot product
            {
                LOG_INFO("Constructing dot product embedding");
                return std::make_unique<DotProduct>(coordinates);
            }
        case CosineEmb:
            // cosine
            {
                LOG_INFO("Constructing cosine embedding");
                return std::make_unique<Cosine>(coordinates);
            }
        case MercatorEmb:
            // mercator
            {
                LOG_INFO("Constructing mercator embedding");
                // split into kappa and rest;
                std::vector<double> kappa;
                std::vector<std::vector<double>> rest;
                std::tie(kappa, rest) = splitFirstColumn(coordinates);
                unused(kappa);

                // one dimensional embedding
                if (rest[0].size() <= 2) {
                    // split into theta and radius
                    std::vector<double> theta;
                    std::vector<double> radius;
                    std::tie(theta, rest) = splitFirstColumn(rest);
                    std::tie(radius, rest) = splitFirstColumn(rest);
                    ASSERT(rest[0].size() == 0);
                    return std::make_unique<MercatorEmbedding>(radius, theta);
                }
                // 2 or more dimensional embedding
                else {
                    ASSERT(rest[0].size() >= 3);
                    // split into radius and rest
                    std::vector<double> radius;
                    std::vector<std::vector<double>> coordinates;
                    std::tie(radius, coordinates) = splitFirstColumn(rest);
                    return std::make_unique<MercatorEmbedding>(radius, coordinates);
                }
            }
        case WeightedNoDimEmb:
            // weighted no dim
            {
                LOG_INFO("Constructing weighted no dim embedding");
                auto pair = splitLastColumn(coordinates);
                return std::make_unique<WeightedNoDim>(pair.first, pair.second);
            }
        case WeightedInfEmb:
            // weighted inf
            {
                LOG_INFO("Constructing weighted inf embedding");
                auto pair = splitLastColumn(coordinates);
                return std::make_unique<WeightedGeometricInf>(pair.first, pair.second);
            }
        case PoincareEmb: {
            LOG_INFO("Constructing poincare embedding");
            return std::make_unique<Poincare>(coordinates);
        }
        default:
            LOG_ERROR("Unknown embedding type");
            return std::unique_ptr<Embedding>(nullptr);
    }
}

std::vector<std::vector<double>> EmbeddingIO::readCoordinatesFromFile(std::string filePath, std::string comment,
                                                                      std::string delimiter) {
    LOG_INFO("Reading coordinates from file: " << filePath);
    std::vector<std::vector<double>> result;

    std::ifstream input(filePath);
    std::string line;

    if (!input.good()) {
        LOG_ERROR("Error while reading file: " << filePath);
        return result;
    }

    // read in the coordinates
    std::map<NodeId, std::vector<double>> coords_dict;
    int coord_size = -1; //dimension of the embedding
    while (std::getline(input, line)) {
        if (line.rfind(comment, 0) == 0) {
            // line starts with comment
            continue;
        }
        std::vector<std::string> tokens = util::splitIntoTokens(line, delimiter);
        NodeId a = std::stoi(tokens[0]);

        std::vector<double> coord(tokens.size() - 1); // dimension of node a
        if (coord_size == -1) {
            coord_size = coord.size();
        } else {
            ASSERT(coord_size == coord.size(), "Problem on line " + line + ": Expected " + std::to_string(coord_size) +
                                                   " coordinates, but got " + std::to_string(coord.size()));
        }

        // pares coordinates and assign to dictionary
        for (int i = 1; i < tokens.size(); i++) {
            coord[i - 1] = std::stod(tokens[i]);
        }
        coords_dict[a] = coord;
    }

    // assert that keys in coords_dict are consecutive starting from 0
    for (NodeId i = 0; i < coords_dict.size(); i++) {
        ASSERT(coords_dict.find(i) != coords_dict.end(), "Node " + std::to_string(i) + " is missing");
    }

    // write coordinates to result
    for (auto& [name, coord] : coords_dict) {
        result.push_back(coord);
    }

    LOG_INFO("Read in " << coords_dict.size() << " coordinates of dimension " << result[0].size());
    input.close();
    return result;
}

std::pair<std::vector<std::vector<double>>, std::vector<double>> EmbeddingIO::splitLastColumn(
    const std::vector<std::vector<double>>& coordinates) {
    std::vector<std::vector<double>> coords(coordinates.size());
    std::vector<double> weights(coordinates.size());

    for (int i = 0; i < coordinates.size(); i++) {
        for (int j = 0; j < coordinates[i].size() - 1; j++) {
            coords[i].push_back(coordinates[i][j]);
        }
        weights[i] = coordinates[i].back();
    }

    return std::make_pair(coords, weights);
}

std::pair<std::vector<double>, std::vector<std::vector<double>>> EmbeddingIO::splitFirstColumn(
    const std::vector<std::vector<double>>& coordinates) {
    std::vector<double> weights(coordinates.size());
    std::vector<std::vector<double>> coords(coordinates.size());

    for (int i = 0; i < coordinates.size(); i++) {
        weights[i] = coordinates[i][0];
        for (int j = 1; j < coordinates[i].size(); j++) {
            coords[i].push_back(coordinates[i][j]);
        }
    }

    return std::make_pair(weights, coords);
}

void EmbeddingIO::writeCoordinates(std::string filePath, const std::vector<std::vector<double>>& positions,
                                   const std::vector<double>& weights) {
    LOG_INFO("Writing coordinates to file " << filePath);
    std::ofstream fil;
    fil.open(filePath);
    fil << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    for (int i = 0; i < positions.size(); i++) {
        fil << i;
        for (int j = 0; j < positions[i].size(); j++) {
            fil << "," << positions[i][j];
        }
        fil << "," << weights[i] << "\n";
    }
    fil.close();
}

void EmbeddingIO::writeCoordinates(std::string filePath, const std::vector<std::vector<double>>& positions) {
    std::ofstream fil;
    fil.open(filePath);
    fil << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    for(int i = 0; i < positions.size(); i++) {
        fil << i;
        for(int j = 0; j < positions[i].size(); j++) {
            fil << "," << positions[i][j];
        }
        fil << "\n";
    }
    fil.close();
}
