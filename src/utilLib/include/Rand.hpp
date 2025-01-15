#pragma once

#include <random>
#include <vector>

class Rand {
   private:
    // implements singleton pattern
    Rand();
    static Rand *instance;
    static Rand *get();  // this returns the singleton but is not needed for the user

    std::mt19937 generator;

   public:
    /**
     * Sets the seed of the random number generator.
     * Otherwise the time of the system will be used
     */
    static void setSeed(int seed);
    /**
     * Random integer between lower and upper bound.
     * The bounds are inclusive
     */
    static int randomInt(int lowerBound, int upperBound);
    static double randomDouble(double lowerBound, double upperBound);
    /**
     * Returns a variable with normal distribution
     * for the given mean and deviation
     */
    static double gaussDistribution(double mean, double deviation);
    /**
     * Random permutation of the numbers 0 to n-1
     */
    static std::vector<int> randomPermutation(int n);

    /**
     * positive random integer
     * represents the number of unsuccessful trials before a first success
     * success has probability prob
     */
    static int geometricVariable(double prob);
};
