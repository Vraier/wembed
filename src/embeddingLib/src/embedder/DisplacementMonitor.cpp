#include "DisplacementMonitor.hpp"

DisplacementMonitor::DisplacementMonitor(double relTol, int patience) : relTol(relTol), patience(patience) {}

void DisplacementMonitor::observe(double relDisplacement) {
    lastRelDisplacement = relDisplacement;
    numObserved++;

    if (relDisplacement < relTol) {
        numSettledSteps++;
    } else {
        numSettledSteps = 0;
    }
}
