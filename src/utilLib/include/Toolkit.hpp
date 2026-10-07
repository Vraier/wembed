#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include <Concepts.hpp>

namespace Toolkit {
std::map<int, int> createIdentity(int max);

/**
 * calculates the smallest and larges number in the input
 */
std::pair<int, int> findMinMax(const std::vector<int>& numbers);

/**
 * calculates the largest and smallest number in numbers and check wether all
 * numbers between occur at least once
 */
bool noGapsInVector(std::vector<int> numbers);

double averageFromVector(const std::vector<double>& values);

/**
 * The pow operation takes a lot of computing time. 
 * We could try to improve this by allowing for less precision
 */
inline flt_t myPow(flt_t base, flt_t exp) {
    return std::pow(base, exp);
}
};  // namespace Toolkit