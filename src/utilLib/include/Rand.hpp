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
    static float randomFloat(float lowerBound, float upperBound);
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
     * Get k random numbers from the range [0, n-1] without replacement
     */
    static std::vector<int> randomSample(int n, int k);
   /**
    * Get k random float coordinates of dimension dim from the range [0, bound]
    */
    static std::vector<std::vector<float>> randomCoordinatesf(int k, int dim, float bound);
    static std::vector<std::vector<double>> randomCoordinates(int k, int dim, double bound);

    /**
     * positive random integer
     * represents the number of unsuccessful trials before a first success
     * success has probability prob
     */
    static int geometricVariable(double prob);
};
