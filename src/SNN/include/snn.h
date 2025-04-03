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
#include <memory>

#include <eigen3/Eigen/Dense>

class SnnModel {
        using Matrix = Eigen::MatrixXd;
        using Vector = Eigen::VectorXd;
        int rows, cols;

        // center and distance to center
        Vector mu, sortVals;
        // normalized and sorted data points
        Matrix normData;

        Vector principal_axis;

        Vector xxt; // for norm computation;

        std::vector<int> sortID;

        // data buffers for queries
        Vector query_buffer, distances_buffer;

    public:
        SnnModel() = default;
        SnnModel(double *data, int r, int c);
        ~SnnModel() = default;

        SnnModel(const SnnModel&) = delete;
        SnnModel& operator=(const SnnModel &) = delete;

        SnnModel(SnnModel&&) = default;
        SnnModel& operator=(SnnModel&&) = default;

        void radius_single_query(double *query, double radius, std::vector<int> *knnID, std::vector<double> *knnDist);
        void radius_batch_query(double *queries, double radius, std::vector<std::vector<int> > *knnID, std::vector<std::vector<double> > *knnDist, 
                                const int qrows);
};
