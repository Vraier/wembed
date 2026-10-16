#include "DVec.hpp"

namespace vectorOperations {

static inline flt_t calculateLPNorm(const CVecRef& x, const CVecRef& y) {
    flt_t sum = 0.0;
    for (size_t i = 0; i < x.dimension(); i++) {
        sum += Toolkit::myPow(std::abs(x[i] - y[i]), flt_t{2});
    }
    return std::sqrt(sum);
}

static inline void differentiateLPNormDifference(const CVecRef& x, const CVecRef& y, const flt_t lpNorm, TmpVec<0>& result) {
    if (lpNorm == flt_t{0.0}) {
        result.setAll(0.0);
        return;
    }

    for (size_t i = 0; i < x.dimension(); i++) {
        const flt_t diff = std::abs(x[i] - y[i]);
        const flt_t sign = (x[i] - y[i]) < flt_t{0} ? -1.0 : 1.0;
        const flt_t derivative = diff / lpNorm * sign;
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
