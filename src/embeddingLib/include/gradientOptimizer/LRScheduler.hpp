#pragma once

#include <memory>

#include "ConvergenceMonitor.hpp"
#include "EmbedderOptions.hpp"

/**
 * Produces the learning rate for each optimization step; the embedder queries it
 * once per step. Iterations are 1-based. learningRate() layers a linear warmup
 * ramp over the first `warmupSteps` on top of the subclass schedule.
 */
class LRScheduler {
   public:
    LRScheduler(float initialRate, int warmupSteps) : initialRate(initialRate), warmupSteps(warmupSteps) {}
    virtual ~LRScheduler() = default;

    float learningRate(int iteration);

   protected:
    virtual float scheduleRate(int iteration) = 0;

    float initialRate;
    int warmupSteps;
};


/**
 * Exponential cooling schedule: LR(t) = initialRate * lrCoolingFactor^t
 * (with linear warmup over the first warmupSteps)
 */
class ExponentialCoolingSchedule : public LRScheduler {
   public:
    ExponentialCoolingSchedule(float initialRate, int warmupSteps, float lrCoolingFactor)
        : LRScheduler(initialRate, warmupSteps), lrCoolingFactor(lrCoolingFactor) {}

   protected:
    float scheduleRate(int iteration) override;

   private:
    float lrCoolingFactor;
};

/**
 * Loss-reactive schedule: a three-zone controller on the monitor's rate(t) with
 * a hysteresis dead zone between the decay and growth thresholds.
 *   rate(t) > lrGrowthThreshold for lrAdaptPatience steps -> LR *= lrGrowthFactor (>=1)
 *   rate(t) < lrDecayThreshold  for lrAdaptPatience steps -> LR *= lrDecayFactor  (<1)
 *   otherwise (dead zone)                                 -> hold
 * Leaving a zone resets its counter, so only sustained behaviour acts. With the
 * default lrGrowthFactor == 1.0 the growth branch is a no-op and this reduces to
 * plateau-decay (ReduceLROnPlateau).
 */
class LossAdaptiveSchedule : public LRScheduler {
   public:
    LossAdaptiveSchedule(float initialRate, int warmupSteps, float lrGrowthFactor, float lrGrowthThreshold,
                         float lrDecayFactor, float lrDecayThreshold, int lrAdaptPatience,
                         const ConvergenceMonitor& monitor)
        : LRScheduler(initialRate, warmupSteps),
          lrGrowthFactor(lrGrowthFactor),
          lrGrowthThreshold(lrGrowthThreshold),
          lrDecayFactor(lrDecayFactor),
          lrDecayThreshold(lrDecayThreshold),
          lrAdaptPatience(lrAdaptPatience),
          monitor(monitor),
          currentRate(initialRate) {}

   protected:
    float scheduleRate(int iteration) override;

   private:
    float lrGrowthFactor;
    float lrGrowthThreshold;
    float lrDecayFactor;
    float lrDecayThreshold;
    int lrAdaptPatience;
    const ConvergenceMonitor& monitor;
    float currentRate;
    int growthSteps = 0;
    int decaySteps = 0;
};

std::unique_ptr<LRScheduler> makeLRScheduler(const EmbedderOptions& opts, const ConvergenceMonitor& monitor);
