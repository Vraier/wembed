#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

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

/**
 * The pow operation takes a lot of computing time. 
 * We could try to improve this by allowing for less precision
 */
inline double myPow(double base, double exp) {
    return std::pow(base, exp);
    // https://martin.ankerl.com/2012/01/25/optimized-approximative-pow-in-c-and-cpp/
    int e = (int)exp;
    union {
        double d;
        int x[2];
    } u = {base};
    u.x[1] = (int)((exp - e) * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;

    // exponentiation by squaring with the exponent's integer part
    // double r = u.d makes everything much slower, not sure why
    double r = 1.0;
    while (e) {
        if (e & 1) {
            r *= base;
        }
        base *= base;
        e >>= 1;
    }

    return r * u.d;
}
};  // namespace Toolkit