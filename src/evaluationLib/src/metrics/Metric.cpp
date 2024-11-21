#include "Metric.hpp"

#include "Macros.hpp"
#include "Rand.hpp"

void Metric::printCSVToConsole(const std::vector<std::string>& content) {
    // output data
    std::string separator;
    std::string result = "";
    for (std::string c : content) {
        result += separator + c;
        separator = ",";
    }
    std::cout << result << std::endl;
}