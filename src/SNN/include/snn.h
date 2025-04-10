/*
MIT License

Copyright (c) 2022 Stefan Güttel, Xinye Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <vector>

#include <eigen3/Eigen/Dense>

class SnnModel {
    public:
        using Matrix = Eigen::MatrixXd;
        using Vector = Eigen::VectorXd;

        SnnModel() = default;
        SnnModel(double *data, int r, int c);
        ~SnnModel() = default;

        SnnModel(const SnnModel&) = delete;
        SnnModel& operator=(const SnnModel &) = delete;

        SnnModel(SnnModel&&) = default;
        SnnModel& operator=(SnnModel&&) = default;

        void radius_single_query(double *query, double radius, std::vector<int>& knnID, Vector& query_buffer, Vector& distance_buffer) const;

        template<typename InputVectorT, typename ResultT, typename ResultMappingFn>
        inline void radius_single_query(const InputVectorT& query, double radius, std::vector<ResultT>& out, ResultMappingFn mapping, Vector& query_buffer, Vector& distance_buffer) const {
            if (query_buffer.size() < cols) {
                query_buffer.resize(cols);
            }
            if (distance_buffer.size() < rows) {
                distance_buffer.resize(rows);
            }

            for (size_t i = 0; i < cols; ++i) {
                query_buffer[i] = query[i] - mu[i];
            }

            auto [left, right] = radius_single_query_impl(query_buffer, radius, distance_buffer);
            radius = radius * radius;

            for (size_t i = left; i < right; i++){
                if (distance_buffer[i] <= radius){
                    out.push_back(mapping(sortID[i]));
                }
            }
        }

    private:
        std::pair<size_t, size_t> radius_single_query_impl(const Vector& normalized_query, double radius, Vector& distances) const;

        int rows, cols;

        // center and distance to center
        Vector mu, sortVals;
        // normalized and sorted data points
        Matrix normData;

        Vector principal_axis;

        Vector xxt; // for norm computation;

        std::vector<int> sortID;
};
