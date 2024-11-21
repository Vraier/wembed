#include "TimeParser.hpp"

#include <fstream>

#include "StringManipulation.hpp"
#include "FileOperations.hpp"

std::vector<std::string> TimeParser::getMetricValues() {
    std::vector<std::string> result;

    if (timePath != "") {
        // read in one line from the time file
        result = util::readLinesFromFile(timePath);
        ASSERT(result.size() == 1, "Time file should contain only one line");
    }
    return result;
}

std::vector<std::string> TimeParser::getMetricNames() {
    std::vector<std::string> result;
    if(timePath != "") {
        result.push_back("embedding_time");
    }
    return result;
}

