#pragma once

#include <algorithm>
#include <cmath>

#include "EmbedderOptions.hpp"

namespace progressEstimate {

// iterations a layer takes until the loss stop fires, -1 for schedules without an estimate.
inline int priorLayerIterations(const EmbedderOptions& opts) {
    const double c = opts.lrCoolingFactor;
    if (opts.lrScheduleType != LRScheduleType::ExponentialCooling || opts.stopCriterion != StopCriterionType::Loss ||
        c <= 0.0 || c >= 1.0 || opts.stopLossTol <= 0.0) {
        return -1;
    }
    const double iterations =
        opts.stopLossPatience + 6.0 * std::pow(std::log(1.0 / c), -0.8) * std::pow(opts.stopLossTol, -0.15);
    return static_cast<int>(std::min(iterations, static_cast<double>(opts.maxIterations)));
}

// -1 while the mean time per iteration is still dominated by the first steps, and once the
// layer has run past its estimate
inline double remainingLayerSeconds(int iteration, int expectedIterations, double layerSeconds) {
    constexpr int minIterations = 30;
    if (iteration < minIterations || expectedIterations <= iteration) {
        return -1.0;
    }
    return (expectedIterations - iteration) * layerSeconds / iteration;
}

}  // namespace progressEstimate
