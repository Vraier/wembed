#pragma once

/**
 * Loss and force factor are defined side by side so they cannot drift apart:
 */
namespace lossFunction {

inline flt_t attractionLoss(flt_t weightedDistance) { return weightedDistance > flt_t{1.0} ? weightedDistance - flt_t{1.0} : 0.0; }

inline flt_t attractionForceFactor(flt_t weightedDistance) { return weightedDistance > flt_t{1.0} ? 1.0 : 0.0; }

// repulsion loss/force are exactly 0 for weightedDistance >= 1
inline flt_t repulsionLoss(flt_t weightedDistance) { return weightedDistance < flt_t{1.0} ? flt_t{1.0} - weightedDistance : 0.0; }

inline flt_t repulsionForceFactor(flt_t weightedDistance) { return weightedDistance < flt_t{1.0} ? 1.0 : 0.0; }

// loss of two identical points, the maximal violation
inline flt_t maxRepulsionLoss() { return repulsionLoss(0.0); }

}  // namespace lossFunction
