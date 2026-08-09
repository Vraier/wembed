#pragma once

#include "Optimizer.hpp"

class SimpleOptimizer : public Optimizer {
   public:
    SimpleOptimizer(int dimension, int numNodes, double maxDisplacement);
    ~SimpleOptimizer();

    void update(VecList& parameters, const VecList& gradients, double learningRate) override;
    void reset() override;

   private:
    int dimension;
    int numNodes;
    double maxDisplacement;

    VecList tmpGradient;
};