#include "ATreeQueries.hpp"

#include <stdexcept>

ATreeQueries::ATreeQueries(const std::vector<std::pair<CVecRef, NodeId>>& points, size_t dimension)
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
        handle_ = atree_create(data.data(), rows, dimension);
    }
}

ATreeQueries::~ATreeQueries() {
    if (handle_) atree_destroy(handle_);
}

ATreeQueries::ATreeQueries(ATreeQueries&& other) noexcept
    : handle_(other.handle_),
      id_translation(std::move(other.id_translation)),
      dimension(other.dimension) {
    other.handle_ = nullptr;
}

ATreeQueries& ATreeQueries::operator=(ATreeQueries&& other) noexcept {
    if (this != &other) {
        if (handle_) atree_destroy(handle_);
        handle_ = other.handle_;
        id_translation = std::move(other.id_translation);
        dimension = other.dimension;
        other.handle_ = nullptr;
    }
    return *this;
}

size_t ATreeQueries::query_sphere(CVecRef point, double radius, std::vector<int>& out) const {
    ASSERT(point.dimension() == dimension);

    if (handle_ && !id_translation.empty()) {
        std::vector<float> query(dimension);
        for (size_t i = 0; i < dimension; ++i) {
            query[i] = static_cast<float>(point[i]);
        }

        uint64_t* ids = nullptr;
        size_t count = 0;
        atree_query_radius_alloc(handle_, query.data(), radius, &ids, &count);

        for (size_t i = 0; i < count; ++i) {
            out.push_back(id_translation[ids[i]]);
        }
        atree_free_results(ids, count);
    }
    return out.size();
}

size_t ATreeQueries::query_nearest(CVecRef, unsigned int, std::vector<int>&) const {
    throw std::runtime_error("Not implemented!");
}

size_t ATreeQueries::query_box(CVecRef, CVecRef, std::vector<int>&) const {
    throw std::runtime_error("Not implemented!");
}
