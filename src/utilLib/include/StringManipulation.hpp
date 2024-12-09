#pragma once

#include <algorithm>
#include <cctype>
#include <locale>
#include <vector>
#include <map>

namespace util {

// trim from both ends (in place)
void trim(std::string &s);
void ltrim(std::string &s);
void rtrim(std::string &s);

// trim from both ends (copying)
std::string trim_copy(std::string s);
std::string ltrim_copy(std::string s);
std::string rtrim_copy(std::string s);

/**
 * Splits a string into tokens. The tokens are separated by the delimiter
*/
std::vector<std::string> splitIntoTokens(std::string &line, std::string delimiter = ",");

/**
 * Can be used to check if a string starts with a comment
*/
bool startsWith(const std::string &s, std::string prefix = "%");

template <typename T>
std::string mapToString(const std::map<T, std::string>& map) {
    std::string result = "";
    for (const auto& [key, value] : map) {
        result += std::to_string(static_cast<int>(key)) + ": " + value + ", ";
    }
    if (!result.empty()) {
        result.pop_back(); // Remove the last space and comma
        result.pop_back();
    }
    return result;
}
};  // namespace util