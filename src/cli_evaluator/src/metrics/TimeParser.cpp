#include "TimeParser.hpp"

#include <fstream>

#include "StringManipulation.hpp"
#include "FileOperations.hpp"

TimeParser::TimeParser(Options opts) : options(opts) {}

std::vector<std::string> TimeParser::getMetricValues() {
    std::vector<std::string> result;

    if (options.timePath != "") {
        // read in one line from the time file
        result = util::readLinesFromFile(options.timePath);
        ASSERT(result.size() == 1, "Time file should contain only one line");
    }
    return result;
}

std::vector<std::string> TimeParser::getMetricNames() {
    std::vector<std::string> result;
    if(options.timePath != "") {
        result.push_back("embedding_time");
    }
    return result;
}

