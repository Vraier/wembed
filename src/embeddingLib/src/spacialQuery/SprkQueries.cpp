#include "SprkQueries.hpp"

#include <stdexcept>

SprkQueries::SprkQueries(const std::vector<std::pair<CVecRef, NodeId>>& points, size_t dimension)
    : handle_(nullptr),
      id_translation(),
      dimension(dimension) {
    ASSERT(dimension >= 2);
    if (!points.empty()) {
        size_t rows = points.size();
        id_translation.reserve(rows);
        std::vector<float> data(rows * dimension);
        for (size_t i = 0; i < rows; ++i) {
            auto [p, id] = points[i];
            ASSERT(p.dimension() == dimension);
            for (size_t j = 0; j < dimension; ++j) {
                data[i * dimension + j] = static_cast<float>(p[j]);
            }
            id_translation.push_back(id);
        }
        handle_ = sprk_create(data.data(), rows, dimension);
    }
}

SprkQueries::~SprkQueries() {
    if (handle_) sprk_destroy(handle_);
}

SprkQueries::SprkQueries(SprkQueries&& other) noexcept
    : handle_(other.handle_),
      id_translation(std::move(other.id_translation)),
      dimension(other.dimension) {
    other.handle_ = nullptr;
}

SprkQueries& SprkQueries::operator=(SprkQueries&& other) noexcept {
    if (this != &other) {
        if (handle_) sprk_destroy(handle_);
        handle_ = other.handle_;
        id_translation = std::move(other.id_translation);
        dimension = other.dimension;
        other.handle_ = nullptr;
    }
    return *this;
}

size_t SprkQueries::query_sphere(CVecRef point, double radius, std::vector<int>& out) const {
    ASSERT(point.dimension() == dimension);

    if (handle_ && !id_translation.empty()) {
        std::vector<float> query(dimension);
        for (size_t i = 0; i < dimension; ++i) {
            query[i] = static_cast<float>(point[i]);
        }

        uint64_t* ids = nullptr;
        size_t count = 0;
        sprk_query_radius(handle_, query.data(), radius, &ids, &count);

        for (size_t i = 0; i < count; ++i) {
            out.push_back(id_translation[ids[i]]);
        }
        sprk_free_results(ids, count);
    }
    return out.size();
}

size_t SprkQueries::query_nearest(CVecRef, unsigned int, std::vector<int>&) const {
    throw std::runtime_error("Not implemented!");
}

size_t SprkQueries::query_box(CVecRef, CVecRef, std::vector<int>&) const {
    throw std::runtime_error("Not implemented!");
}
