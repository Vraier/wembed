#pragma once

#include <vector>
#include <memory>
#include <omp.h>

#include "Graph.hpp"
#include "VecList.hpp"
#include "SpacialIndex.hpp"
#include "snn.h"


class SNNQueries: public SpatialIndex {
   public:
    using IdType = int;

    SNNQueries(const std::vector<std::pair<CVecRef, NodeId>>& points, size_t dimension);

    size_t query_sphere(CVecRef point, double radius, std::vector<int>& out) const override;

    // note: no support for box or kNN queries
    size_t query_nearest(CVecRef point, unsigned int number, std::vector<int>& out) const override;
    size_t query_box(CVecRef minCorner, CVecRef maxCorner, std::vector<int>& out) const override;

   private:
    SnnModel snn;
    std::vector<IdType> result_buffer;
    std::vector<NodeId> id_translation;
    size_t dimension;

    // thread-local buffers
    // (yes, this is a bit of an ugly hack, but anything else requires serious changes to the SpatialIndex interface)
    static SnnModel::Vector query_buffer;
    #pragma omp threadprivate(query_buffer)
    static SnnModel::Vector distance_buffer;
    #pragma omp threadprivate(distance_buffer)
};

