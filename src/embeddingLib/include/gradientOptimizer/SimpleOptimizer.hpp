#pragma once

#include "Optimizer.hpp"

class SimpleOptimizer : public Optimizer {
   public:
    SimpleOptimizer(int dimension, int numNodes, double learningRate, double coolingFactor, double maxDisplacement);
    ~SimpleOptimizer();

    void update(VecList& parameters, const VecList& gradients) override;
    void reset() override;

   private:
    int dimension;
    int numNodes;
    double learningRate;
    double coolingFactor;
    double maxDisplacement;

    VecList tmpGradient;
    int t;      // Time step
};