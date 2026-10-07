#include "FileOperations.hpp"

#include <fstream>
#include <stdexcept>


namespace util {

std::vector<std::string> readLinesFromFile(std::string pathToFile) {
    std::vector<std::string> lines;

    // check if file exists
    std::ifstream input(pathToFile);
    if (!input.good()) {
        throw std::runtime_error("could not open file " + pathToFile);
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