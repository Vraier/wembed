#pragma once

#include "Optimizer.hpp"

class SimpleOptimizer : public Optimizer {
   public:
    SimpleOptimizer(int dimension, int numNodes, float maxDisplacement);
    ~SimpleOptimizer();

    void update(VecList<>& parameters, const VecList<>& gradients, float learningRate) override;
    void reset() override;

   private:
    int dimension;
    int numNodes;
    float maxDisplacement;

    VecList<> tmpGradient;
};