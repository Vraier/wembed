#pragma once

#include "Metric.hpp"
#include "Options.hpp"

/**
 * This class reads the time file and returns the time it took to embed the graph.
 * The time file contains a single line with the time in seconds.
 */
class TimeParser : public Metric {
   public:
    TimeParser(Options options);
    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();

   private:
    Options options;
};