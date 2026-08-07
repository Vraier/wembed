#include "SprkQueries.hpp"

#include <stdexcept>

SprkQueries::SprkQueries(const std::vector<std::pair<CVecRef, NodeId>>& points, size_t dimension)
    : handle_(nullptr),
      dimension(dimension) {
    ASSERT(dimension >= 2);
    if (!points.empty()) {
        size_t rows = points.size();
        std::vector<float> data(rows * dimension);
        for (size_t i = 0; i < rows; ++i) {
            auto [p, id] = points[i];
            ASSERT(p.dimension() == dimension);
            ASSERT(id == i);
            for (size_t j = 0; j < dimension; ++j) {
                data[i * dimension + j] = static_cast<float>(p[j]);
            }
        }
        handle_ = sprk_create(data.data(), rows, dimension);
    }
}

SprkQueries::~SprkQueries() {
    if (handle_) sprk_destroy(handle_);
}

SprkQueries::SprkQueries(SprkQueries&& other) noexcept
    : handle_(other.handle_),
      dimension(other.dimension) {
    other.handle_ = nullptr;
}

SprkQueries& SprkQueries::operator=(SprkQueries&& other) noexcept {
    if (this != &other) {
        if (handle_) sprk_destroy(handle_);
        handle_ = other.handle_;
        dimension = other.dimension;
        other.handle_ = nullptr;
    }
    return *this;
}

size_t SprkQueries::query_sphere(CVecRef point, double radius, std::vector<uint64_t>& out) const {
    ASSERT(point.dimension() == dimension);

    if (handle_) {
        std::vector<float> query(dimension);
        for (size_t i = 0; i < dimension; ++i) {
            query[i] = static_cast<float>(point[i]);
        }

        uint64_t* ids = nullptr;
        size_t count = 0;
        sprk_query_radius(handle_, query.data(), radius, &ids, &count);

        out.resize(count);
        for (size_t i = 0; i < count; ++i) {
            out[i] = ids[i];
        }
        sprk_free_results(ids, count);
    }
    return out.size();
}

size_t SprkQueries::query_nearest(CVecRef, unsigned int, std::vector<uint64_t>&) const {
    throw std::runtime_error("Not implemented!");
}

size_t SprkQueries::query_box(CVecRef, CVecRef, std::vector<uint64_t>&) const {
    throw std::runtime_error("Not implemented!");
}
