#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "DVec.hpp"

class SpatialIndex {
   public:
    virtual ~SpatialIndex() = default;

    // Query for points within a certain radius from a point (range query).
    // Returned ids are positions in the point array the index was built from.
    virtual size_t query_sphere(CVecRef point, double radius, std::vector<uint64_t>& out) const = 0;
};
