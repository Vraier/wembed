#pragma once

#include <cmath>
#include <limits>
#include <memory>

#include "ConvergenceMonitor.hpp"
#include "EmbedderOptions.hpp"
#include "Toolkit.hpp"

/**
 * Produces the learning rate for each optimization step. The embedder queries
 * the schedule once per step and passes the result to the optimizer.
 * Iterations are 1-based: the first optimization step queries iteration 1.
 */
class LRScheduler {
   public:
    LRScheduler(double initialRate, int warmupSteps) : initialRate(initialRate), warmupSteps(warmupSteps) {}
    virtual ~LRScheduler() = default;

    // schedule value with the warmup ramp applied on top
    double learningRate(int iteration) {
        const double lr = scheduleRate(iteration);
        if (iteration < warmupSteps) {
            return lr * static_cast<double>(iteration) / static_cast<double>(warmupSteps);
        }
        return lr;
    }

   protected:
    virtual double scheduleRate(int iteration) = 0;

    double initialRate;
    int warmupSteps;
};

class ExponentialCoolingSchedule : public LRScheduler {
   public:
    ExponentialCoolingSchedule(double initialRate, int warmupSteps, double coolingFactor)
        : LRScheduler(initialRate, warmupSteps), coolingFactor(coolingFactor) {}

   protected:
    double scheduleRate(int iteration) override {
        return initialRate * Toolkit::myPow(coolingFactor, static_cast<double>(iteration));
    }

   private:
    double coolingFactor;
};

/**
 * Loss-reactive schedule. A step that improves the best observed loss by more
 * than growthRelTol (relative) grows the rate by growthFactor. Otherwise, once
 * the loss has stagnated for another plateauPatience steps (the monitor's
 * counter, margin stopRelTol), the rate drops by decayFactor. Growing only on
 * new bests keeps the rate from ratcheting up while the loss oscillates
 * without real progress. A growthFactor of 1 disables growth, leaving a pure
 * plateau-decay behavior. Growth tracks its own best loss because the monitor
 * only ratchets its best when the (possibly different) stopRelTol margin is
 * beaten.
 */
class LossAdaptiveSchedule : public LRScheduler {
   public:
    LossAdaptiveSchedule(double initialRate, int warmupSteps, double growthFactor, double growthRelTol,
                         double decayFactor, int plateauPatience, const ConvergenceMonitor& monitor)
        : LRScheduler(initialRate, warmupSteps),
          growthFactor(growthFactor),
          growthRelTol(growthRelTol),
          decayFactor(decayFactor),
          plateauPatience(plateauPatience),
          monitor(monitor),
          currentRate(initialRate) {}

   protected:
    double scheduleRate(int /*iteration*/) override {
        const int stagnant = monitor.stagnantSteps();
        // the monitor's counter was reset by a significant improvement since the last cut
        if (stagnant < lastCutStagnation) {
            lastCutStagnation = 0;
        }

        // one new observation per step; the very first one only initializes the best
        bool improvedSignificantly = false;
        if (monitor.numObservations() > numSeenObservations) {
            numSeenObservations = monitor.numObservations();
            const double loss = monitor.lastLoss();
            if (loss < bestLoss - growthRelTol * std::abs(bestLoss)) {
                improvedSignificantly = numSeenObservations >= 2;
                bestLoss = loss;
            }
        }

        if (growthFactor != 1.0 && improvedSignificantly) {
            currentRate *= growthFactor;
        } else if (stagnant - lastCutStagnation >= plateauPatience) {
            currentRate *= decayFactor;
            lastCutStagnation = stagnant;
        }
        return currentRate;
    }

   private:
    double growthFactor;
    double growthRelTol;
    double decayFactor;
    int plateauPatience;
    const ConvergenceMonitor& monitor;
    double currentRate;
    int lastCutStagnation = 0;
    double bestLoss = std::numeric_limits<double>::max();
    int numSeenObservations = 0;
};

inline std::unique_ptr<LRScheduler> makeLRScheduler(const EmbedderOptions& opts, const ConvergenceMonitor& monitor) {
    switch (opts.lrScheduleType) {
        case LRScheduleType::ExponentialCooling:
            return std::make_unique<ExponentialCoolingSchedule>(opts.learningRate, opts.warmupSteps,
                                                                opts.coolingFactor);
        case LRScheduleType::LossAdaptive:
            return std::make_unique<LossAdaptiveSchedule>(opts.learningRate, opts.warmupSteps, opts.growthFactor,
                                                          opts.growthRelTol, opts.decayFactor, opts.plateauPatience,
                                                          monitor);
    }
    return std::make_unique<ExponentialCoolingSchedule>(opts.learningRate, opts.warmupSteps, opts.coolingFactor);
}
