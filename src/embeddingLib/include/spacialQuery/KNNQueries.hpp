#pragma once

#include <vector>
#include <memory>
#include <omp.h>

#include "Graph.hpp"
#include "VecList.hpp"
#include "SpacialIndex.hpp"
#include "SNNQueries.hpp"


class KNNQueries: public SpatialIndex {
   public:
    using IdType = int;
    using Vec = std::vector<float>;
    using Dataset = std::vector<Vec>;

    // Similarity function: Negative squared L2 distance
    struct NegL2 {
        using value_type = float;
        float operator()(const Vec& a, const Vec& b) const {
            float sum = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                float diff = a[i] - b[i];
                sum -= diff * diff;
            }
            return sum;
        }
    };

    KNNQueries(const std::vector<std::pair<CVecRef, NodeId>>& points, size_t dimension);

    size_t query_sphere(CVecRef point, double radius, std::vector<int>& out) const override;
    size_t query_nearest(CVecRef point, unsigned int number, std::vector<int>& out) const override;
    size_t query_box(CVecRef minCorner, CVecRef maxCorner, std::vector<int>& out) const override;

   private:

};

