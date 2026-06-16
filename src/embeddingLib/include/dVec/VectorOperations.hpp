#include "DVec.hpp"

namespace vectorOperations {

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
}
