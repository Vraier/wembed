#pragma once

#include <algorithm>
#include <cmath>

#include "EmbedderOptions.hpp"

namespace progressEstimate {

// iterations a layer takes until the loss stop fires, -1 for schedules without an estimate.
inline int priorLayerIterations(const EmbedderOptions& opts) {
    const flt_t c = opts.lrCoolingFactor;
    if (opts.lrScheduleType != LRScheduleType::ExponentialCooling || opts.stopCriterion != StopCriterionType::Loss ||
        c <= flt_t{0.0} || c >= flt_t{1.0} || opts.stopLossTol <= flt_t{0.0}) {
        return -1;
    }
    const flt_t iterations =
        static_cast<flt_t>(opts.stopLossPatience) + flt_t{6.0} * std::pow(std::log(flt_t{1.0} / c), flt_t{-0.8}) * std::pow(opts.stopLossTol, flt_t{-0.15});
    return static_cast<int>(std::min(iterations, static_cast<flt_t>(opts.maxIterations)));
}

// -1 while the mean time per iteration is still dominated by the first steps, and once the
// layer has run past its estimate
inline flt_t remainingLayerSeconds(int iteration, int expectedIterations, flt_t layerSeconds) {
    constexpr int minIterations = 30;
    if (iteration < minIterations || expectedIterations <= iteration) {
        return -1.0;
    }
    return static_cast<flt_t>(expectedIterations - iteration) * layerSeconds / static_cast<flt_t>(iteration);
}

}  // namespace progressEstimate
