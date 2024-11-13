#include "FileOperations.hpp"

#include <fstream>

#include "Macros.hpp"

namespace util {

std::vector<std::string> readLinesFromFile(std::string pathToFile) {
    std::vector<std::string> lines;

    // check if file exists
    std::ifstream input(pathToFile);
    if (!input.good()) {
        LOG_ERROR( "Error while reading file: " << pathToFile);
        return lines;
    }

    // read in the lines
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }
    input.close();
    return lines;
}

};  // namespace util