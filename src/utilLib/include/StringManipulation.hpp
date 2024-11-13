#pragma once

#include <algorithm>
#include <cctype>
#include <locale>
#include <vector>

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
 * @param line the string to split
 * @param delimiter the delimiter to use
 * @return a vector of tokens
*/
std::vector<std::string> splitIntoTokens(std::string &line, std::string delimiter = ",");

/**
 * Can be used to check if a string starts with a comment
*/
bool startsWith(const std::string &s, std::string prefix = "%");
};  // namespace util