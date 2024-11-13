#pragma once

#include "EvalOptions.hpp"
#include "Graph.hpp"
#include "Metric.hpp"
#include "VecList.hpp"

class ConfigParser : public Metric {
   public:
    ConfigParser(OptionValues options);
    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();

   private:
    std::vector<std::string> extractMetricsByRegex(std::string pathToLogFile, std::string regex, int position);

    inline static const std::string embedderRegex = "> ([^\\(\\)=]+)(\\(default\\))?=(.*)";
    inline static const std::string node2VecRegex = ".*\\(-(.*)\\)=(.*)";

    OptionValues options;
};