#include "DisplacementMonitor.hpp"

DisplacementMonitor::DisplacementMonitor(flt_t relTol, int patience) : relTol(relTol), patience(patience) {}

void DisplacementMonitor::observe(flt_t relDisplacement) {
    lastRelDisplacement = relDisplacement;
    numObserved++;

    if (relDisplacement < relTol) {
        numSettledSteps++;
    } else {
        numSettledSteps = 0;
    }
}
