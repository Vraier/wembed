#include "SNNQueries.hpp"
#include <iostream>


SNNQueries::SNNQueries(const std::vector<std::pair<CVecRef, NodeId>>& points, size_t dimension):
    snn(),
    dist_buffer(),
    input_buffer(),
    id_translation(),
    dimension(dimension) {
        ASSERT(dimension >= 1);
        if (!points.empty()) {
            size_t rows = points.size();
            id_translation.reserve(rows);
            double* data = new double[rows * dimension];
            for (size_t i = 0; i < rows; ++i) {
                auto [p, id] = points[i];
                ASSERT(p.dimension() == dimension);
                for (size_t j = 0; j < dimension; ++j) {
                    // the SNN data structure is row major
                    data[i + j * rows] = p[j];
                }
                id_translation.push_back(id);
            }
            snn = SnnModel(data, rows, dimension);
            input_buffer.reserve(dimension);
            delete[] data;
        }
    }


size_t SNNQueries::query_sphere(CVecRef point, double radius, std::vector<int>& out) {
    ASSERT(point.dimension() == dimension);
    if (!id_translation.empty()) {
        result_buffer.clear();
        dist_buffer.clear();
        input_buffer.clear();
        input_buffer.resize(dimension);
        for (size_t j = 0; j < dimension; ++j) {
            input_buffer[j] = point[j];
        }

        snn.radius_single_query(input_buffer.data(), radius, &result_buffer, &dist_buffer);
        for (IdType id: result_buffer) {
            out.push_back(id_translation[id]);
        }
    }
    return out.size();
}
