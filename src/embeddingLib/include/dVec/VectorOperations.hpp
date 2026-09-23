#include "DVec.hpp"

namespace vectorOperations {

    /*
     *  LP Norm computations
     */

static inline double calculateLPNorm(const CVecRef& x, const CVecRef& y) {
    double sum = 0.0;
    for (size_t i = 0; i < x.dimension(); i++) {
        sum += Toolkit::myPow(std::abs(x[i] - y[i]), 2);
    }
    return std::sqrt(sum);
}

static inline void differentiateLPNormDifference(const CVecRef& x, const CVecRef& y, const double lpNorm, TmpVec<0>& result) {
    if (lpNorm == 0.0) {
        result.setAll(0.0);
        return;
    }

    for (size_t i = 0; i < x.dimension(); i++) {
        const double diff = std::abs(x[i] - y[i]);
        const double sign = (x[i] - y[i]) < 0 ? -1.0 : 1.0;
        const double derivative = diff / lpNorm * sign;
        result[i] = derivative;
    }
}

/**
 * Given x and y, calculate sigma/sigma x ||x-y||_p
 */
static inline void differentiateLPNormDifference(const CVecRef& x, const CVecRef& y, TmpVec<0>& result) {
    differentiateLPNormDifference(x, y, calculateLPNorm(x, y), result);
}

    /*
     * Dot product norm computations
     */

static inline double calculateDotProductNorm(const CVecRef& x, const CVecRef& y) {
    double sum = 0.0;
    for (int i = 0; i < x.dimension(); i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

/*
 * Given x and y, computes sigma/sigma_x <x,y>
 */
static inline void differentiateDotProductNorm([[maybe_unused]]const CVecRef& x, const CVecRef& y,
                                               [[maybe_unused]]double norm, TmpVec<0>& result) {
    result = y;
}

static inline void differentiateDotProductNorm(const CVecRef& x, const CVecRef& y, TmpVec<0>& result) {
    differentiateDotProductNorm(x, y, calculateDotProductNorm(x, y), result);
}
}
