#pragma once

#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "Timings.hpp"
#include "VecList.hpp"

class WEmbedEmbedder : public EmbedderInterface {
    using Timer = util::Timer;

   public:
    virtual ~WEmbedEmbedder() {};

    virtual void calculateStep();
    virtual bool isFinished();
    virtual void calculateEmbedding();

    virtual std::vector<std::vector<double>> getCoordinates() = 0;
    virtual std::vector<double> getWeights();

    virtual void setCoordinates(const std::vector<std::vector<double>> &coordinates);
    virtual void setWeights(const std::vector<double> &weights) = 0;

   private:
    virtual void repulstionForce(int v, int u);
    virtual void attractionForce(int v, int u);

    virtual void calculateAllAttractingForces();
    virtual void calculateAllRepellingForces();

    /**
     * Calculates the (weighted) distance between the two nodes given the current weights and positions.
     */
    virtual double calculateDistance(int v, int u);

    Timer timer;
    EmbedderOptions options;

    int currentIteration = 0;

    VecList currentForce;
    VecList currentPositions;
    std::vector<double> currentWeights;  // currently not changed during gradient descent
};