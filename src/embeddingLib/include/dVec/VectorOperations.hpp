#include "DVec.hpp"

namespace vectorOperations {

static inline double calculateLPNorm(const CVecRef x, const CVecRef y, int p = 2) {
    double sum = 0.0;
    for (size_t i = 0; i < x.dimension(); i++) {
        sum += Toolkit::myPow(std::abs(x[i] - y[i]), p);
    }
    return Toolkit::myPow(sum, 1.0 / p);
}

static inline void differentiateLPNormDifference(const CVecRef x, const CVecRef y, const double lpNorm, TmpVec<0>& result, int p = 2) {
    if (lpNorm == 0.0) {
        result.setAll(0.0);
        return;
    }

    for (size_t i = 0; i < x.dimension(); i++) {
        double diff = std::abs(x[i] - y[i]);
        double sign = (x[i] - y[i]) < 0 ? -1.0 : 1.0;
        double derivative = Toolkit::myPow(diff / lpNorm, p-1) * sign;
        result[i] = derivative;
    }
}

/**
 * Given x and y, calculate sigma/sigma x ||x-y||_p
 */
static inline void differentiateLPNormDifference(const CVecRef x, const CVecRef y, TmpVec<0>& result, int p = 2) {
    differentiateLPNormDifference(x, y, calculateLPNorm(x, y, p), result, p);
}
}
