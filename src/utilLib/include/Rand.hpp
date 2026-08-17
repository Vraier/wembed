#pragma once

#include <cstdint>
#include <random>
#include <vector>

class Rand {
   private:
    // implements singleton pattern
    Rand();
    static Rand *instance;
    static Rand *get();  // this returns the singleton but is not needed for the user

    std::mt19937 generator;
    // base seed the singleton was seeded with; used to derive independent
    // deterministic generators (see localGenerator)
    uint32_t seedValue = 0;

   public:
    /**
     * Sets the seed of the random number generator.
     * Otherwise the time of the system will be used
     */
    static void setSeed(int seed);

    /**
     * Reference to the shared singleton generator. Only safe to use
     * single-threaded (it is the default source for the non-parallel RNG helpers).
     */
    static std::mt19937 &globalGenerator();

    /**
     * Returns a fresh, independent generator seeded deterministically from the
     * global base seed and the two key components (e.g. node id and iteration).
     * Use this instead of the shared singleton inside parallel regions.
     */
    static std::mt19937 localGenerator(uint32_t a, uint32_t b);
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
