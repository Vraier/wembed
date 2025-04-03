#pragma once

#include <vector>

#include "Graph.hpp"
#include "VecList.hpp"
#include "SpacialIndex.hpp"
#include "snn.h"


// TODO: we could also get the distances from here, so we don't need to calculate them a second time
class SNNQueries: public SpatialIndex {
   public:
    using IdType = int;

    SNNQueries(const std::vector<std::pair<CVecRef, NodeId>>& points, size_t dimension);

    size_t query_sphere(CVecRef point, double radius, std::vector<int>& out);

   private:
    SnnModel snn;
    std::vector<IdType> result_buffer;
    std::vector<double> dist_buffer;
    std::vector<double> input_buffer;
    std::vector<NodeId> id_translation;
    size_t dimension;
};

