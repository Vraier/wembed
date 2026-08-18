#include "DisplacementMonitor.hpp"

DisplacementMonitor::DisplacementMonitor(float relTol, int patience) : relTol(relTol), patience(patience) {}

void DisplacementMonitor::observe(float relDisplacement) {
    lastRelDisplacement = relDisplacement;
    numObserved++;

    if (relDisplacement < relTol) {
        numSettledSteps++;
    } else {
        numSettledSteps = 0;
    }
}
