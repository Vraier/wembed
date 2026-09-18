#include "SprkQueries.hpp"

SprkQueries::SprkQueries(const std::vector<CVecRef>& points, const size_t dimension)
    : handle_(nullptr),
      dimension(dimension) {
    ASSERT(dimension >= 2);
    if (!points.empty()) {
        const size_t rows = points.size();
        std::vector<float> data(rows * dimension);
        for (size_t i = 0; i < rows; ++i) {
            auto p = points[i];
            ASSERT(p.dimension() == dimension);
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
    out.clear();

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
