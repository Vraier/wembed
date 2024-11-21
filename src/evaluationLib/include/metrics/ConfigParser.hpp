#pragma once

#include "Graph.hpp"
#include "Metric.hpp"
#include "VecList.hpp"

enum class LogType { None = 0, WEmbed = 1, CSV = 2 };

/**
 * Parses a log file of an embedder and extracts useful information.
 * It can read the logfile from WEmbed and also a simple CSV format. 
 */
class ConfigParser : public Metric {
   public:
    ConfigParser(std::string logPath, LogType logType) : logPath(logPath), logType(logType) {}
    std::vector<std::string> getMetricValues();
    std::vector<std::string> getMetricNames();

   private:
    std::vector<std::string> extractMetricsByRegex(std::string pathToLogFile, std::string regex, int position);

    inline static const std::string embedderRegex = "> ([^\\(\\)=]+)(\\(default\\))?=(.*)";
    inline static const std::string node2VecRegex = ".*\\(-(.*)\\)=(.*)";

    std::string logPath;
    LogType logType;
};