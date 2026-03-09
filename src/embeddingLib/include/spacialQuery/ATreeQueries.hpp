#pragma once

#include <vector>

#include "Graph.hpp"
#include "VecList.hpp"
#include "SpacialIndex.hpp"
#include "atree.h"

class ATreeQueries : public SpatialIndex {
   public:
    ATreeQueries(const std::vector<std::pair<CVecRef, NodeId>>& points, size_t dimension);
    ~ATreeQueries() override;

    // Move-only (handle cannot be shared)
    ATreeQueries(ATreeQueries&& other) noexcept;
    ATreeQueries& operator=(ATreeQueries&& other) noexcept;
    ATreeQueries(const ATreeQueries&) = delete;
    ATreeQueries& operator=(const ATreeQueries&) = delete;

    size_t query_sphere(CVecRef point, double radius, std::vector<int>& out) const override;
    size_t query_nearest(CVecRef point, unsigned int number, std::vector<int>& out) const override;
    size_t query_box(CVecRef minCorner, CVecRef maxCorner, std::vector<int>& out) const override;

   private:
    ATreeHandle* handle_;
    std::vector<NodeId> id_translation;
    size_t dimension;
};
