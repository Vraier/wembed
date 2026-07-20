#pragma once

#include <vector>

#include "Graph.hpp"
#include "VecList.hpp"
#include "SpacialIndex.hpp"
#include "sprk.h"

class SprkQueries : public SpatialIndex {
   public:
    SprkQueries(const std::vector<std::pair<CVecRef, NodeId>>& points, size_t dimension);
    ~SprkQueries() override;

    // Move-only (handle cannot be shared)
    SprkQueries(SprkQueries&& other) noexcept;
    SprkQueries& operator=(SprkQueries&& other) noexcept;
    SprkQueries(const SprkQueries&) = delete;
    SprkQueries& operator=(const SprkQueries&) = delete;

    size_t query_sphere(CVecRef point, double radius, std::vector<int>& out) const override;
    size_t query_nearest(CVecRef point, unsigned int number, std::vector<int>& out) const override;
    size_t query_box(CVecRef minCorner, CVecRef maxCorner, std::vector<int>& out) const override;

   private:
    SprkHandle* handle_;
    std::vector<NodeId> id_translation;
    size_t dimension;
};