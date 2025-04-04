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

#include <cassert>
#include <iostream>
#include <eigen3/Eigen/SVD>

void svd_eigen_sovler(const Matrix& mat, Matrix& vt) {
    Eigen::BDCSVD<Eigen::MatrixXd> svd_wrapper = mat.bdcSvd(Eigen::ComputeFullV);
    assert(!svd_wrapper.computeU() && svd_wrapper.computeV());
    auto info = svd_wrapper.info();
    if (info == Eigen::ComputationInfo::Success) {
        vt = svd_wrapper.matrixV();
    } else {
        // this is bad, just choose any axis
        std::cout << "Warning: Singular Value Decomposition failed" << std::endl;
        vt(0, 0) = 1.0;
    }
}
