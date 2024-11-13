#pragma once

#include "Metric.hpp"

class TimeParser : public Metric {
   public:
    TimeParser(OptionValues options);
    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();

   private:
    OptionValues options;
};