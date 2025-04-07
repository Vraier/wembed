#include "Rand.hpp"
#include "Macros.hpp"

#include <unordered_set>

Rand* Rand::instance = nullptr;  // or NULL, or nullptr in c++11

Rand::Rand() {
    // i think this makes a random seed every time (if system supports random device)
    std::random_device device;
    generator = std::mt19937(device());
}

Rand* Rand::get() {
    if (instance == nullptr) {
        instance = new Rand();
    }
    return instance;
}

void Rand::setSeed(int seed) { get()->generator = std::mt19937(seed); }

int Rand::randomInt(int lowerBound, int upperBound) {
    std::uniform_int_distribution<int> distribution(lowerBound, upperBound);
    return distribution(get()->generator);
}

double Rand::randomDouble(double lowerBound, double upperBound) {
    std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
    return distribution(get()->generator);
}

double Rand::gaussDistribution(double mean, double deviation) {
    std::normal_distribution<double> distribution{mean, deviation};
    return distribution(get()->generator);
}

std::vector<int> Rand::randomPermutation(int n) {
    std::vector<int> result(n);
    for (int i = 0; i < n; i++) {
        result[i] = i;
    }

    for (int i = n - 1; i > 0; --i) {
        int j = randomInt(0, i);
        // swap a[i] and a[j]
        int tmp = result[i];
        result[i] = result[j];
        result[j] = tmp;
    }

    return result;
}

std::vector<int> Rand::randomSample(int n, int k) {
    ASSERT(n >= k, "Sample size k cannot be larger than population size n");
    ASSERT(k >= 0, "Sample size k must be positive");
    
    // https://www.nowherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html
    std::unordered_set<int> selected;
    for(int r = n-k; r < n; r++){
        int v = randomInt(0, r);
        if(!selected.insert(v).second){
            selected.insert(r);
        }
    }

    std::vector<int> result(selected.begin(), selected.end());
    return result;
}

int Rand::geometricVariable(double prob) {
    std::geometric_distribution<int> distribution(prob);
    return distribution(get()->generator);
}