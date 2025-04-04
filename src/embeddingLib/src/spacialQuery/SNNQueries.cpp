#include "SNNQueries.hpp"

// initialize statics
SnnModel::Vector SNNQueries::query_buffer;
SnnModel::Vector SNNQueries::distance_buffer;

SNNQueries::SNNQueries(const std::vector<std::pair<CVecRef, NodeId>>& points, size_t dimension):
    snn(),
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
            delete[] data;
        }
    }


size_t SNNQueries::query_sphere(CVecRef point, double radius, std::vector<int>& out) const {
    ASSERT(point.dimension() == dimension);

    if (!id_translation.empty()) {
        snn.radius_single_query(point, radius, out, [&](int id){ return id_translation[id]; }, query_buffer, distance_buffer);
        for (IdType id: result_buffer) {
            out.push_back(id_translation[id]);
        }
    }
    return out.size();
}


size_t SNNQueries::query_nearest(CVecRef, unsigned int, std::vector<int>&) const {
    throw std::runtime_error("Not implemented!");
}

size_t SNNQueries::query_box(CVecRef, CVecRef, std::vector<int>&) const {
    throw std::runtime_error("Not implemented!");
}
