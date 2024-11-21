#include "ConfigParser.hpp"

#include <fstream>
#include <regex>

#include "StringManipulation.hpp"
#include "FileOperations.hpp"

std::vector<std::string> ConfigParser::getMetricValues() {
    std::vector<std::string> result;
    std::vector<std::string> tmp;

    if (logPath != "") {
        switch (logType) {
            case LogType::WEmbed:
                tmp = extractMetricsByRegex(logPath, embedderRegex, 3);
                break;
            case LogType::CSV:
                tmp = util::splitIntoTokens(util::readLinesFromFile(logPath)[1]);
                break;
            default:
                LOG_ERROR("Unknown config type");
                break;
        }
    }
    result.insert(result.end(), tmp.begin(), tmp.end());
    return result;
}

std::vector<std::string> ConfigParser::getMetricNames() {
    std::vector<std::string> result;
    std::vector<std::string> tmp;

    if (logPath != "") {
        switch (logType) {
            case LogType::WEmbed:
                tmp = extractMetricsByRegex(logPath, embedderRegex, 1);
                break;
            case LogType::CSV:
                tmp = util::splitIntoTokens(util::readLinesFromFile(logPath)[0]);
                break;
            default:
                LOG_ERROR("Unknown config type");
                break;
        }
    }
    result.insert(result.end(), tmp.begin(), tmp.end());
    return result;
}

std::vector<std::string> ConfigParser::extractMetricsByRegex(std::string pathToLogFile, std::string regex, int position) {
    std::vector<std::string> lines;
    std::vector<std::string> result;

    std::ifstream input(pathToLogFile);
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }
    input.close();

    std::regex configLine{regex};
    std::smatch m;

    for (std::string line : lines) {
        if (std::regex_match(line, m, configLine)) {
            result.push_back(m[position]);
        }
    }

    return result;
}