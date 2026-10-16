#include <gtest/gtest.h>

#include "LossFunction.hpp"
#include "test_macros.hpp"
#include "wembed_types.hpp"

using wembed::flt_t;

namespace {

constexpr flt_t H = std::is_same_v<float, flt_t> ? 1e-3 : 1e-7;

flt_t numDerivative(flt_t (*f)(flt_t), flt_t wd) { return (f(wd + H) - f(wd - H)) / (2 * H); }

// wd grid avoiding the kink at the threshold, where the numerical derivative
// straddles both pieces
std::vector<flt_t> gridAvoidingKink() {
    std::vector<flt_t> grid;
    for (flt_t wd = 0.05; wd < flt_t{3.0}; wd += flt_t{0.037}) {
        if (std::abs(wd - flt_t{1.0}) < 2 * H) continue;
        grid.push_back(wd);
    }
    return grid;
}

}  // namespace

TEST(LossFunction, IsExactHinge) {
    for (flt_t wd = 0.0; wd < flt_t{3.0}; wd += flt_t{0.01}) {
        EXPECT_FLT_T_EQ(lossFunction::attractionLoss(wd), std::max(flt_t{0.0}, wd - flt_t{1.0}));
        EXPECT_FLT_T_EQ(lossFunction::repulsionLoss(wd), std::max(flt_t{0.0}, flt_t{1.0} - wd));
    }
}

TEST(LossFunction, ForceFactorIsLossDerivative) {
    constexpr flt_t err = std::is_same_v<float, flt_t> ? 1e-3 : 1e-5;
    for (flt_t wd : gridAvoidingKink()) {
        EXPECT_NEAR(lossFunction::attractionForceFactor(wd),
                    numDerivative(lossFunction::attractionLoss, wd), err)
            << "attraction, wd=" << wd;
        EXPECT_NEAR(lossFunction::repulsionForceFactor(wd),
                    -numDerivative(lossFunction::repulsionLoss, wd), err)
            << "repulsion, wd=" << wd;
    }
}

// the spatial index query radius relies on repulsion vanishing beyond the threshold
TEST(LossFunction, RepulsionSupportEndsAtThreshold) {
    for (flt_t wd = 1.0; wd < flt_t{3.0}; wd += flt_t{0.01}) {
        EXPECT_EQ(lossFunction::repulsionLoss(wd), flt_t{0.0});
        EXPECT_EQ(lossFunction::repulsionForceFactor(wd), flt_t{0.0});
    }
    for (flt_t wd = 0.0; wd <= flt_t{1.0}; wd += flt_t{0.01}) {
        EXPECT_EQ(lossFunction::attractionLoss(wd), flt_t{0.0});
        EXPECT_EQ(lossFunction::attractionForceFactor(wd), flt_t{0.0});
    }
}

TEST(LossFunction, MaxRepulsionLossMatchesCoincidentPair) {
    EXPECT_FLT_T_EQ(lossFunction::maxRepulsionLoss(), flt_t{1.0});
    EXPECT_FLT_T_EQ(lossFunction::maxRepulsionLoss(), lossFunction::repulsionLoss(flt_t{0.0}));
}
