#pragma once


class WeightedIndex {
    IndexType type;
    int embeddingDimension;
    std::shared_ptr<SpatialIndex> index = nullptr;

public:
    WeightedIndex(const IndexType type, const int embeddingDimension) : type(type), embeddingDimension(embeddingDimension) {

    }

    void updateIndex (std::vector<std::pair<CVecRef, NodeId>> points) {
        switch (type) {
            case IndexType::Sprk:
                index = std::make_shared<SprkQueries>(std::move(points), embeddingDimension);
                break;
            default:
                LOG_ERROR("Unknown Index Type");
                break;
        }
    }

    void querySphere(CVecRef position, const double weight, const double radius, std::vector<NodeId> output) const {

        const double queryRadius = radius * Toolkit::myPow(weight * weight, 1.0 / static_cast<double>(embeddingDimension));

        ASSERT(position.dimension() == embeddingDimension);
        ASSERT(queryRadius > 0);

        index->query_sphere(position, queryRadius, output);
    }
};