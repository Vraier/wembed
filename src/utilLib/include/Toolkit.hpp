#pragma once

#include <algorithm>
#include <vector>
#include <map>

class Toolkit {
   public:

    static std::map<int, int> createIdentity(int max);

    /**
     * calculates the smallest and larges number in the input
     */
    static std::pair<int, int> findMinMax(const std::vector<int>& numbers);

    /**
     * calculates the largest and smallest number in numbers and check wether all
     * numbers between occur at least once
     */
    static bool noGapsInVector(std::vector<int> numbers);
};