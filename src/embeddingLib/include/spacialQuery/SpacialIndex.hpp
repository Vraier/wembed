#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "DVec.hpp"

class SpatialIndex {
   public:
    virtual ~SpatialIndex() = default;

    // Query for nearest neighbors (point-based query)
    virtual size_t query_nearest(CVecRef point, unsigned int number, std::vector<int>& out) const = 0;

    // Query for points within a certain radius from a point (range query)
    virtual size_t query_sphere(CVecRef point, double radius, std::vector<int>& out) const = 0;

    // Query for points in a box (range query)
    virtual size_t query_box(CVecRef minCorner, CVecRef maxCorner, std::vector<int>& out) const = 0;

    // Check if the index is empty
    // virtual bool is_empty() const = 0;

    // Get the number of elements in the index
    // virtual size_t size() const = 0;
};