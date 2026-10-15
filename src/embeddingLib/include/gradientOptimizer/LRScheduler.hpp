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
    LRScheduler(flt_t initialRate, int warmupSteps) : initialRate(initialRate), warmupSteps(warmupSteps) {}
    virtual ~LRScheduler() = default;

    flt_t learningRate(int iteration);

   protected:
    virtual flt_t scheduleRate(int iteration) = 0;

    flt_t initialRate;
    int warmupSteps;
};


/**
 * Exponential cooling schedule: LR(t) = initialRate * lrCoolingFactor^t
 * (with linear warmup over the first warmupSteps)
 */
class ExponentialCoolingSchedule : public LRScheduler {
   public:
    ExponentialCoolingSchedule(flt_t initialRate, int warmupSteps, flt_t lrCoolingFactor)
        : LRScheduler(initialRate, warmupSteps), lrCoolingFactor(lrCoolingFactor) {}

   protected:
    flt_t scheduleRate(int iteration) override;

   private:
    flt_t lrCoolingFactor;
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
    LossAdaptiveSchedule(flt_t initialRate, int warmupSteps, flt_t lrGrowthFactor, flt_t lrGrowthThreshold,
                         flt_t lrDecayFactor, flt_t lrDecayThreshold, int lrAdaptPatience,
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
    flt_t scheduleRate(int iteration) override;

   private:
    flt_t lrGrowthFactor;
    flt_t lrGrowthThreshold;
    flt_t lrDecayFactor;
    flt_t lrDecayThreshold;
    int lrAdaptPatience;
    const ConvergenceMonitor& monitor;
    flt_t currentRate;
    int growthSteps = 0;
    int decaySteps = 0;
};

std::unique_ptr<LRScheduler> makeLRScheduler(const EmbedderOptions& opts, const ConvergenceMonitor& monitor);
