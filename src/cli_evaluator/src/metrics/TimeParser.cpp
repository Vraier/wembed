#include "TimeParser.hpp"

#include <fstream>

#include "StringManipulation.hpp"
#include "FileOperations.hpp"

TimeParser::TimeParser(OptionValues opts) : options(opts) {}

std::vector<std::string> TimeParser::getMetricValues() {
    std::vector<std::string> result;

    if (options.timeFile != "") {
        // read in one line from the time file
        result = util::readLinesFromFile(options.timeFile);
        ASSERT(result.size() == 1, "Time file should contain only one line");
    }
    return result;
}

std::vector<std::string> TimeParser::getMetricNames() {
    std::vector<std::string> result;
    if(options.timeFile != "") {
        result.push_back("embedding_time");
    }
    return result;
}

