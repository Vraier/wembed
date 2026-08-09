#pragma once

#include <cmath>
#include <limits>

/**
 * Tracks the loss across steps and decides when it has stagnated.
 * An observation counts as an improvement when the loss undercuts the best
 * observed value by a relative margin of relTol. After patience many
 * non-improving steps in a row the loss is considered converged.
 */
class ConvergenceMonitor {
   public:
    ConvergenceMonitor(double relTol, int patience) : relTol(relTol), patience(patience) {}

    void observe(double loss) {
        lastObservedLoss = loss;
        numObserved++;

        if (loss < bestLoss - relTol * std::abs(bestLoss)) {
            bestLoss = loss;
            numStagnantSteps = 0;
        } else {
            numStagnantSteps++;
        }
    }

    bool converged() const { return numStagnantSteps >= patience; }

    // number of consecutive steps without significant improvement
    int stagnantSteps() const { return numStagnantSteps; }

    // loss history for loss-reactive schedules
    int numObservations() const { return numObserved; }
    double lastLoss() const { return lastObservedLoss; }

   private:
    double relTol;
    int patience;
    double bestLoss = std::numeric_limits<double>::max();
    int numStagnantSteps = 0;
    int numObserved = 0;
    double lastObservedLoss = 0.0;
};
