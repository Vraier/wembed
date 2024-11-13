#include "StringManipulation.hpp"

namespace util {

// trim from start (in place)
void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

// trim from both ends (in place)
void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
}

// trim from start (copying)
std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

std::vector<std::string> splitIntoTokens(std::string &line, std::string delimiter) {
    std::vector<std::string> result;

    while (line.size()) {
        int index = line.find(delimiter);
        if (index != std::string::npos) {
            result.push_back(line.substr(0, index));
            line = line.substr(index + delimiter.size());
            if (line.size() == 0) result.push_back(line);
        } else {
            result.push_back(line);
            line = "";
        }
    }
    return result;
}

bool startsWith(const std::string &s, std::string prefix) { return s.rfind(prefix, 0) == 0; }

};  // namespace util