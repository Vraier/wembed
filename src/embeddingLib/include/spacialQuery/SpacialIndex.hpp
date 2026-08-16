#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "DVec.hpp"

class SpatialIndex {
   public:
    virtual ~SpatialIndex() = default;

    // Query for points within a certain radius from a point (range query)
    virtual size_t query_sphere(CVecRef point, double radius, std::vector<int>& out) const = 0;
};