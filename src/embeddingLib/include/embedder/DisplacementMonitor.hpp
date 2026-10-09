#pragma once
#include "Concepts.hpp"

/**
 * Tracks the per-step relative node displacement (mean node movement / radius of
 * gyration, so scale- and dimension-invariant) and reports the layout settled
 * once it stays below relTol for `patience` steps in a row.
 */
class DisplacementMonitor {
   public:
    DisplacementMonitor(flt_t relTol, int patience);

    void observe(flt_t relDisplacement);

    bool converged() const { return numSettledSteps >= patience; }
    int settledSteps() const { return numSettledSteps; }
    int numObservations() const { return numObserved; }
    flt_t lastDisplacement() const { return lastRelDisplacement; }

   private:
    flt_t relTol;
    int patience;
    int numSettledSteps = 0;
    int numObserved = 0;
    flt_t lastRelDisplacement = 0.0;
};
