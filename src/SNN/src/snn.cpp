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


#include "snn.h"

#include <cmath>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <cassert>

#include "eign.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Eigen::seqN;
using Eigen::placeholders::all;
using RawVec = Eigen::Map<Eigen::VectorXd>;

void argsort(const Vector& input, std::vector<int>& output) {
    std::iota(output.begin(), output.end(), 0);
    std::sort(output.begin(), output.end(),
              [&input](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return input[left] < input[right];
              });
}

void reorderArray1D(const Vector& input, Vector& output, const std::vector<int>& index) {
    assert(input.size() == output.size());
    for (size_t i = 0; i < index.size(); ++i) {
        output[i] = input[index[i]];
    }
}

void reorderArray2D(const Matrix& input, Matrix& output, const std::vector<int>& index) {
    assert(input.rows() == output.rows() && input.cols() == output.cols());
    for (size_t i = 0; i < input.rows(); ++i) {
        output(i, all) = input(index[i], all);
    }
}

void calculate_matrix_mean(const Matrix& mat, Vector& ret) {
    for (size_t i = 0; i < mat.cols(); i++){
        ret[i] = mat(all, i).sum() / static_cast<double>(mat.rows());
    }
}

void calculate_skip_euclid_norm(const Vector& xxt, const Matrix& mat, const Vector& arr, Vector& ret, size_t start, size_t end) {
    const auto range = seqN(start, end-start);

    double inner_prod = arr.dot(arr);
    ret(range) = xxt(range).array() + inner_prod;
    ret(range) -= 2.0 * mat(range, all) * arr;
}

// for 1-dimensional data
size_t binarySearch(const Vector& arr, double point){
    size_t lo = 0, hi = arr.size();

    while (hi != lo) {
        size_t mid = (hi + lo) / 2;
        if (arr[mid] < point) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}


SnnModel::SnnModel(double *data, int r, int c): rows(r), cols(c) {
    Matrix tempNormData(rows, cols);
    normData.resize(rows, cols);
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            tempNormData(r, c) = data[r + rows * c];
        }
    }

    sortVals.resize(rows);
    Vector temp_sortVals(rows);

    Matrix vt(cols, cols);
    principal_axis.resize(cols);

    // standardize data
    mu.resize(cols);
    calculate_matrix_mean(tempNormData, mu);
    for (size_t i = 0; i < rows; ++i) {
        tempNormData(i, all) -= mu;
    }

    // singular value decomposition, obtain the sort_values
    if (cols > 1) { 
        svd_eigen_sovler(tempNormData, vt);
        principal_axis = vt(0, all);

        // TODO: zero would be bad?!
        double sign_flip = (principal_axis[0] > 0) ? 1 : ((principal_axis[0] < 0) ? -1 : 0); // flip sign
        principal_axis *= sign_flip;
        temp_sortVals = tempNormData * principal_axis;
    } else if (cols == 1){
        principal_axis[0] = 1.0;
        temp_sortVals = tempNormData(all, 0);
    } else{
        std::cerr << "Error occured in input, please enter correct value for cols." << std::endl;
        std::exit(1);
    }

    // order data by distance to center
    sortID.resize(rows);
    argsort(temp_sortVals, sortID);
    reorderArray1D(temp_sortVals, sortVals, sortID);
    reorderArray2D(tempNormData, normData, sortID);

    // precompute distances to center
    xxt.resize(rows);
    for (size_t i = 0; i < normData.rows(); ++i) {
        xxt[i] = normData(i, all).dot(normData(i, all));
    }
}

void SnnModel::radius_single_query(double *query, double radius, std::vector<int>& knnID, Vector& query_buffer, Vector& distance_buffer) const {
    radius_single_query(RawVec(query, cols), radius, knnID, [](int id){ return id; }, query_buffer, distance_buffer);
}

std::pair<size_t, size_t> SnnModel::radius_single_query_impl(const Vector& normalized_query, double radius, Vector& distances) const {
    double sv_q = principal_axis.dot(normalized_query);
    size_t left = binarySearch(sortVals, sv_q-radius);
    size_t right = binarySearch(sortVals, sv_q+radius);
    calculate_skip_euclid_norm(xxt, normData, normalized_query, distances, left, right);

    return {left, right};
}
