#pragma once

#include <vector>

#include "Graph.hpp"
#include "EvalOptions.hpp"
#include "VecList.hpp"

class StressError {
   public:
    /**
     * This needs to calculate all pair shortest pats
     */
    static std::vector<std::string> getMetricValues(const OptionValues &options, const Graph &g, const VecList &coords);
    static std::vector<std::string> getMetricNames();

   private:
    static double calculateOptimalFullStressScale(const OptionValues &options, const Graph &g, const VecList &coords, const std::vector<std::vector<int>> &allDist);
    static double calculateOptimalMaxentStressScale(const OptionValues &options, const Graph &g, const VecList &coords);

    static double calculateFullStress(const OptionValues &options, const Graph &g, const VecList &coords, const std::vector<std::vector<int>> &allDist, double scale);
    static double calculateMaxEntStress(const OptionValues &options, const Graph &g, const VecList &coords, double scale);

    static constexpr double maxentMinAlpha = 0.008;
};