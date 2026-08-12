#pragma once

#include <limits>
#include <vector>

/**
 * Tracks the EMA-smoothed training loss and exposes the windowed relative loss
 * decrease
 *     rate(t) = (Lbar(t - rateWindow) - Lbar(t)) / max(|Lbar(t - rateWindow)|, tiny)
 * The loss has converged once rate(t) stays below relTol for `patience`
 * consecutive steps. Until a full window is buffered, rate(t)
 * reads STILL_IMPROVING so neither the stop nor a loss-reactive LR schedule
 * reacts during that warmup.
 */
class ConvergenceMonitor {
   public:
    static constexpr double STILL_IMPROVING = std::numeric_limits<double>::infinity();

    ConvergenceMonitor(double relTol, int patience, double smoothingFactor, int rateWindow);

    void observe(double loss);

    bool converged() const { return numStagnantSteps >= patience; }
    int stagnantSteps() const { return numStagnantSteps; }
    double relImprovement() const { return lastRate; }  // rate(t); STILL_IMPROVING during warmup
    int numObservations() const { return numObserved; }
    double lastLoss() const { return lastObservedLoss; }

   private:
    static constexpr double TINY = 1e-12;

    double relTol;
    int patience;
    double smoothingFactor;
    int rateWindow;

    std::vector<double> ring;  // last rateWindow + 1 smoothed losses
    int ringHead = 0;          // next write slot == oldest retained sample once full
    int ringCount = 0;

    double smoothedLoss = 0.0;
    int numObserved = 0;
    double lastObservedLoss = 0.0;
    double lastRate = STILL_IMPROVING;
    int numStagnantSteps = 0;
};
