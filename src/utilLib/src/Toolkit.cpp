#include "Toolkit.hpp"

std::map<int, int> Toolkit::createIdentity(int max) {
    std::map<int, int> identity;
    for (int i = 0; i < max; i++) {
        identity[i] = i;
    }
    return identity;
}

std::pair<int, int> Toolkit::findMinMax(const std::vector<int>& numbers) {
    int smallest = *std::min_element(numbers.begin(), numbers.end());
    int largest = *std::max_element(numbers.begin(), numbers.end());
    return std::make_pair(smallest, largest);
}

bool Toolkit::noGapsInVector(std::vector<int> numbers) {
    std::pair<int, int> minMax = findMinMax(numbers);
    int smallest = minMax.first;
    int largest = minMax.second;

    if (largest - smallest + 1 > numbers.size()) {
        return false;  // Range too large to fit in seen vector
    }

    std::vector<bool> seen(largest - smallest + 1, false);
    for (int num : numbers) {
        if (num >= smallest && num <= largest) {
            seen[num - smallest] = true;
        }
    }
    for (bool val : seen) {
        if (!val) {
            return false;
        }
    }
    return true;
}

double Toolkit::averageFromVector(const std::vector<double>& values) {
    double sum = 0;
    for (double val : values) {
        sum += val;
    }
    if (values.size() == 0) {
        return -1;
    } else {
        return sum / (double)values.size();
    }
}
