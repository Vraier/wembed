#include <gtest/gtest.h>

#include "LossFunction.hpp"

namespace {

constexpr double H = 1e-7;

double numDerivative(double (*f)(double), double wd) { return (f(wd + H) - f(wd - H)) / (2 * H); }

// wd grid avoiding the kink at the threshold, where the numerical derivative
// straddles both pieces
std::vector<double> gridAvoidingKink() {
    std::vector<double> grid;
    for (double wd = 0.05; wd < 3.0; wd += 0.037) {
        if (std::abs(wd - 1.0) < 2 * H) continue;
        grid.push_back(wd);
    }
    return grid;
}

}  // namespace

TEST(LossFunction, IsExactHinge) {
    for (double wd = 0.0; wd < 3.0; wd += 0.01) {
        EXPECT_DOUBLE_EQ(lossFunction::attractionLoss(wd), std::max(0.0, wd - 1.0));
        EXPECT_DOUBLE_EQ(lossFunction::repulsionLoss(wd), std::max(0.0, 1.0 - wd));
    }
}

TEST(LossFunction, ForceFactorIsLossDerivative) {
    for (double wd : gridAvoidingKink()) {
        EXPECT_NEAR(lossFunction::attractionForceFactor(wd),
                    numDerivative(lossFunction::attractionLoss, wd), 1e-5)
            << "attraction, wd=" << wd;
        EXPECT_NEAR(lossFunction::repulsionForceFactor(wd),
                    -numDerivative(lossFunction::repulsionLoss, wd), 1e-5)
            << "repulsion, wd=" << wd;
    }
}

// the spatial index query radius relies on repulsion vanishing beyond the threshold
TEST(LossFunction, RepulsionSupportEndsAtThreshold) {
    for (double wd = 1.0; wd < 3.0; wd += 0.01) {
        EXPECT_EQ(lossFunction::repulsionLoss(wd), 0.0);
        EXPECT_EQ(lossFunction::repulsionForceFactor(wd), 0.0);
    }
    for (double wd = 0.0; wd <= 1.0; wd += 0.01) {
        EXPECT_EQ(lossFunction::attractionLoss(wd), 0.0);
        EXPECT_EQ(lossFunction::attractionForceFactor(wd), 0.0);
    }
}

TEST(LossFunction, MaxRepulsionLossMatchesCoincidentPair) {
    EXPECT_DOUBLE_EQ(lossFunction::maxRepulsionLoss(), 1.0);
    EXPECT_DOUBLE_EQ(lossFunction::maxRepulsionLoss(), lossFunction::repulsionLoss(0.0));
}
