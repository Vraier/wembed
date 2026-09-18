#pragma once

#include <vector>

#include "Graph.hpp"
#include "VecList.hpp"
#include "SpacialIndex.hpp"
#include "sprk.h"

class SprkQueries : public SpatialIndex {
   public:
    SprkQueries(const std::vector<CVecRef>& points, size_t dimension);
    ~SprkQueries() override;

    // Move-only (handle cannot be shared)
    SprkQueries(SprkQueries&& other) noexcept;
    SprkQueries& operator=(SprkQueries&& other) noexcept;
    SprkQueries(const SprkQueries&) = delete;
    SprkQueries& operator=(const SprkQueries&) = delete;

    size_t query_sphere(CVecRef point, double radius, std::vector<uint64_t>& out) const override;

   private:
    SprkHandle* handle_;
    size_t dimension;
};
