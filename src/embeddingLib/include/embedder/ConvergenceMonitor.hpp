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
    static constexpr float STILL_IMPROVING = std::numeric_limits<float>::infinity();

    ConvergenceMonitor(float relTol, int patience, float smoothingFactor, int rateWindow);

    void observe(float loss);

    bool converged() const { return numStagnantSteps >= patience; }
    int stagnantSteps() const { return numStagnantSteps; }
    float relImprovement() const { return lastRate; }  // rate(t); STILL_IMPROVING during warmup
    int numObservations() const { return numObserved; }
    float lastLoss() const { return lastObservedLoss; }

   private:
    static constexpr float TINY = 1e-12;

    float relTol;
    int patience;
    float smoothingFactor;
    int rateWindow;

    std::vector<float> ring;  // last rateWindow + 1 smoothed losses
    int ringHead = 0;          // next write slot == oldest retained sample once full
    int ringCount = 0;

    float smoothedLoss = 0.0;
    int numObserved = 0;
    float lastObservedLoss = 0.0;
    float lastRate = STILL_IMPROVING;
    int numStagnantSteps = 0;
};
