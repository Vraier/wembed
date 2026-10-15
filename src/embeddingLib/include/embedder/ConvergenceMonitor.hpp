#pragma once

#include <limits>
#include <vector>

#include <Concepts.hpp>

/**
 * Tracks the EMA-smoothed training loss and exposes the windowed relative loss
 * decrease
 *     rate(t) = (Lbar(t - rateWindow) - Lbar(t)) / max(|Lbar(t - rateWindow)|, tiny)
 * The loss has converged once |rate(t)| stays below relTol for `patience`
 * consecutive steps. Until a full window is buffered, rate(t)
 * reads STILL_IMPROVING so neither the stop nor a loss-reactive LR schedule
 * reacts during that warmup.
 */
class ConvergenceMonitor {
   public:
    static constexpr flt_t STILL_IMPROVING = std::numeric_limits<flt_t>::infinity();

    // lossFloor: lower bound of the rate's denominator. Below it the tolerance acts on the
    // absolute loss change, so a (nearly) perfectly embedded graph with loss ~ 0 still stops
    ConvergenceMonitor(flt_t relTol, int patience, flt_t smoothingFactor, int rateWindow, flt_t lossFloor = TINY);

    void observe(flt_t loss);

    bool converged() const { return numStagnantSteps >= patience; }
    int stagnantSteps() const { return numStagnantSteps; }
    flt_t relImprovement() const { return lastRate; }  // rate(t); STILL_IMPROVING during warmup
    int numObservations() const { return numObserved; }
    flt_t lastLoss() const { return lastObservedLoss; }

   private:
    static constexpr flt_t TINY = 1e-12;

    flt_t relTol;
    flt_t lossFloor;
    int patience;
    flt_t smoothingFactor;
    int rateWindow;

    std::vector<flt_t> ring;  // last rateWindow + 1 smoothed losses
    int ringHead = 0;          // next write slot == oldest retained sample once full
    int ringCount = 0;

    flt_t smoothedLoss = 0.0;
    int numObserved = 0;
    flt_t lastObservedLoss = 0.0;
    flt_t lastRate = STILL_IMPROVING;
    int numStagnantSteps = 0;
};
