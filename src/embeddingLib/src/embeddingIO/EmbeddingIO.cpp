#include "EmbeddingIO.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "Cosine.hpp"
#include "DotProduct.hpp"
#include "Euclidean.hpp"
#include "InfNorm.hpp"
#include "GraphIO.hpp"
#include "MercatorEmbedding.hpp"
#include "Poincare.hpp"
#include "StringManipulation.hpp"
#include "WeightedGeometric.hpp"
#include "WeightedGeometricInf.hpp"
#include "WeightedNoDim.hpp"
#include "Additive.hpp"

namespace {
std::ofstream openForWriting(const std::string& filePath) {
    std::ofstream fil(filePath);
    if (!fil.is_open()) {
        throw std::runtime_error("could not open " + filePath + " for writing");
    }
    return fil;
}
}  // namespace

std::unique_ptr<Embedding> EmbeddingIO::parseEmbedding(EmbeddingType type, const std::vector<std::vector<double>>& coordinates, int lpNorm) {
    switch (type) {
        case WeightedEmb:
            // weighted
            {
                auto pair = splitLastColumn(coordinates);
                return std::make_unique<WeightedGeometric>(pair.first, pair.second, lpNorm);
            }

        case EuclideanEmb:
            // euclidean
            {
                return std::make_unique<Euclidean>(coordinates);
            }
        case DotProductEmb:
            // dot product
            {
                return std::make_unique<DotProduct>(coordinates);
            }
        case CosineEmb:
            // cosine
            {
                return std::make_unique<Cosine>(coordinates);
            }
        case MercatorEmb:
            // mercator
            {
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
                auto pair = splitLastColumn(coordinates);
                return std::make_unique<WeightedNoDim>(pair.first, pair.second);
            }
        case WeightedInfEmb:
            // weighted inf
            {
                auto pair = splitLastColumn(coordinates);
                return std::make_unique<WeightedGeometricInf>(pair.first, pair.second);
            }
        case PoincareEmb: {
            return std::make_unique<Poincare>(coordinates);
        }
        case InfNormEmb: {
            return std::make_unique<InfNorm>(coordinates);
        }
        case AdditiveEmb: {
            auto pair = splitLastColumn(coordinates);
            return std::make_unique<Additive>(pair.first, pair.second);
        }
        default:
            throw std::invalid_argument("unknown embedding type " + std::to_string(type));
    }
}

std::vector<std::vector<double>> EmbeddingIO::readCoordinatesFromFile(std::string filePath, std::string comment,
                                                                      std::string delimiter) {
    std::ifstream input(filePath);
    if (!input.good()) {
        throw std::runtime_error("could not open coordinate file " + filePath);
    }

    // read in the coordinates
    std::map<NodeId, std::vector<double>> coords_dict;
    int coord_size = -1; //dimension of the embedding
    std::string line;
    int lineNumber = 0;
    while (std::getline(input, line)) {
        lineNumber++;
        if (line.rfind(comment, 0) == 0 || line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }
        const auto invalidLine = [&](const std::string& reason) {
            return std::invalid_argument(filePath + ":" + std::to_string(lineNumber) + ": " + reason);
        };

        // splitIntoTokens consumes its argument
        std::string rest = line;
        std::vector<std::string> tokens = util::splitIntoTokens(rest, delimiter);
        std::vector<double> coord(tokens.size() - 1); // dimension of node a
        if (coord_size == -1) {
            coord_size = coord.size();
        } else if (coord_size != static_cast<int>(coord.size())) {
            throw invalidLine("expected " + std::to_string(coord_size) + " coordinates, got " +
                              std::to_string(coord.size()));
        }

        NodeId a = 0;
        try {
            a = std::stoi(tokens[0]);
            for (size_t i = 1; i < tokens.size(); i++) {
                coord[i - 1] = std::stod(tokens[i]);
            }
        } catch (const std::logic_error&) {
            throw invalidLine("invalid number");
        }
        coords_dict[a] = coord;
    }

    // ids have to be consecutive starting from 0
    for (NodeId i = 0; i < static_cast<NodeId>(coords_dict.size()); i++) {
        if (coords_dict.find(i) == coords_dict.end()) {
            throw std::invalid_argument(filePath + ": vertex " + std::to_string(i) + " is missing");
        }
    }

    std::vector<std::vector<double>> result;
    for (auto& [name, coord] : coords_dict) {
        result.push_back(coord);
    }
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
    std::ofstream fil = openForWriting(filePath);
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
    std::ofstream fil = openForWriting(filePath);
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
