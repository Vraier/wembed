#pragma once

/**
 * Tracks the per-step relative node displacement (mean node movement / radius of
 * gyration, so scale- and dimension-invariant) and reports the layout settled
 * once it stays below relTol for `patience` steps in a row.
 */
class DisplacementMonitor {
   public:
    DisplacementMonitor(float relTol, int patience);

    void observe(float relDisplacement);

    bool converged() const { return numSettledSteps >= patience; }
    int settledSteps() const { return numSettledSteps; }
    int numObservations() const { return numObserved; }
    float lastDisplacement() const { return lastRelDisplacement; }

   private:
    float relTol;
    int patience;
    int numSettledSteps = 0;
    int numObserved = 0;
    float lastRelDisplacement = 0.0;
};
