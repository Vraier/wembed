#include "LRScheduler.hpp"

#include <memory>

#include "Toolkit.hpp"

double LRScheduler::learningRate(int iteration) {
    const double lr = scheduleRate(iteration);
    if (iteration < warmupSteps) {
        return lr * static_cast<double>(iteration) / static_cast<double>(warmupSteps);
    }
    return lr;
}

double ExponentialCoolingSchedule::scheduleRate(int iteration) {
    return initialRate * Toolkit::myPow(lrCoolingFactor, static_cast<double>(iteration));
}

double LossAdaptiveSchedule::scheduleRate(int /*iteration*/) {
    const double rate = monitor.relImprovement();

    if (rate > lrGrowthThreshold) {
        decaySteps = 0;
        if (++growthSteps >= lrAdaptPatience) {
            currentRate *= lrGrowthFactor;
            growthSteps = 0;
        }
    } else if (rate < lrDecayThreshold) {
        growthSteps = 0;
        if (++decaySteps >= lrAdaptPatience) {
            currentRate *= lrDecayFactor;
            decaySteps = 0;
        }
    } else {
        growthSteps = 0;  // dead zone: hold, reset both counters
        decaySteps = 0;
    }
    return currentRate;
}

std::unique_ptr<LRScheduler> makeLRScheduler(const EmbedderOptions& opts, const ConvergenceMonitor& monitor) {
    switch (opts.lrScheduleType) {
        case LRScheduleType::ExponentialCooling:
            return std::make_unique<ExponentialCoolingSchedule>(opts.learningRate, opts.warmupSteps,
                                                                opts.lrCoolingFactor);
        case LRScheduleType::LossAdaptive:
            return std::make_unique<LossAdaptiveSchedule>(opts.learningRate, opts.warmupSteps, opts.lrGrowthFactor,
                                                          opts.lrGrowthThreshold, opts.lrDecayFactor,
                                                          opts.lrDecayThreshold, opts.lrAdaptPatience, monitor);
    }
    return std::make_unique<ExponentialCoolingSchedule>(opts.learningRate, opts.warmupSteps, opts.lrCoolingFactor);
}
