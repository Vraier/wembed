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


#include "eign.h"
#include "snn.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <cassert>

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
    // else if (arr[sortID[hi]] == point) {
    //     return hi;
    // }
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

    query_buffer.resize(cols);
    distances_buffer.resize(rows);
}


// query
void SnnModel::radius_single_query(double *query, double radius, std::vector<int> *knnID, std::vector<double> *knnDist){
    RawVec query_vec(query, cols);
    query_buffer = query_vec - mu;

    double sv_q = principal_axis.dot(query_buffer);
    size_t left = binarySearch(sortVals, sv_q-radius);
    size_t right = binarySearch(sortVals, sv_q+radius);
    calculate_skip_euclid_norm(xxt, normData, query_buffer, distances_buffer, left, right);

    radius = pow(radius, 2);

    (*knnID).clear();
    // (*knnDist).clear();

    for (size_t i = left; i < right; i++){
        if (distances_buffer[i] <= radius){
            (*knnID).push_back(sortID[i]);
            // (*knnDist).push_back(sqrt(distances_buffer[i]));
        }
    }
}



// batch helper functions
void extract_sample(double *queries, double *query, const int num, const int *rows, const int *cols){ // columns major order
    for (int i=0; i<*cols; i++){
        *(query + i) = *(queries + *rows * i + num);
    }
}

void insert_vector(std::vector<std::vector<int> > *knnID, std::vector<std::vector<double> > *knnDist, 
                        std::vector<int> *knnID_unit, std::vector<double> *knnDist_unit, int i, int qcols){

    for (int j=0; j<qcols; j++){
        (*knnID)[i][j] = (*knnID_unit)[j];
        (*knnDist)[i][j] = (*knnDist_unit)[j];
    }
}

// query in batch
void SnnModel::radius_batch_query(double *queries, double radius, std::vector<std::vector<int> > *knnID, 
                                      std::vector<std::vector<double> > *knnDist, const int qrows){
    double query[cols];
    std::vector<int> knnID_unit;
    std::vector<double> knnDist_unit;


    (*knnID).clear();
    (*knnDist).clear();

    (*knnID).resize(qrows);
    (*knnDist).resize(qrows);

    // #pragma omp parallel for
    // ---> keep it single-threaded for now
    for (int i=0; i<qrows; i++){
        extract_sample(queries, query, i, &qrows, &cols);
        this->radius_single_query(query, radius, &knnID_unit, &knnDist_unit);

        (*knnID)[i].resize(knnID_unit.size());
        (*knnDist)[i].resize(knnDist_unit.size());
        insert_vector(knnID, knnDist, &knnID_unit, &knnDist_unit, i, knnID_unit.size());
    }
        
}
