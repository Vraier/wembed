#pragma once

/**
 * Loss and force factor are defined side by side so they cannot drift apart:
 */
namespace lossFunction {

inline double attractionLoss(double weightedDistance) { return weightedDistance > 1.0 ? weightedDistance - 1.0 : 0.0; }

inline double attractionForceFactor(double weightedDistance) { return weightedDistance > 1.0 ? 1.0 : 0.0; }

// repulsion loss/force are exactly 0 for weightedDistance >= 1
inline double repulsionLoss(double weightedDistance) { return weightedDistance < 1.0 ? 1.0 - weightedDistance : 0.0; }

inline double repulsionForceFactor(double weightedDistance) { return weightedDistance < 1.0 ? 1.0 : 0.0; }

// loss of two idential points, the maximal violation
inline double maxRepulsionLoss() { return repulsionLoss(0.0); }

}  // namespace lossFunction
