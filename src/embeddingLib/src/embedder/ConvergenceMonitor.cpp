#include "ConvergenceMonitor.hpp"

#include <algorithm>
#include <cmath>

ConvergenceMonitor::ConvergenceMonitor(float relTol, int patience, float smoothingFactor, int rateWindow)
    : relTol(relTol),
      patience(patience),
      smoothingFactor(smoothingFactor),
      rateWindow(rateWindow < 1 ? 1 : rateWindow),
      // rateWindow intervals span rateWindow + 1 samples (endpoints included)
      ring(static_cast<std::size_t>(this->rateWindow) + 1, 0.0) {}

void ConvergenceMonitor::observe(float loss) {
    if (numObserved == 0) {
        smoothedLoss = loss;
    } else {
        smoothedLoss = smoothingFactor * loss + (1.0f - smoothingFactor) * smoothedLoss;
    }
    lastObservedLoss = smoothedLoss;
    numObserved++;

    ring[ringHead] = smoothedLoss;
    ringHead = (ringHead + 1) % ring.size();
    if (ringCount < static_cast<int>(ring.size())) {
        ringCount++;
    }

    if (ringCount >= static_cast<int>(ring.size())) {
        const float windowStart = ring[ringHead];  // Lbar(t - rateWindow)
        const float denom = std::max(std::abs(windowStart), TINY);
        lastRate = (windowStart - smoothedLoss) / denom;
    } else {
        lastRate = STILL_IMPROVING;
    }

    if (lastRate < relTol) {
        numStagnantSteps++;
    } else {
        numStagnantSteps = 0;
    }
}
