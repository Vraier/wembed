#pragma once

#include "Optimizer.hpp"

class SimpleOptimizer : public Optimizer {
   public:
    SimpleOptimizer(int dimension, int numNodes, flt_t maxDisplacement);
    ~SimpleOptimizer();

    void update(VecList<>& parameters, const VecList<>& gradients, flt_t learningRate) override;
    void reset() override;

   private:
    int dimension;
    int numNodes;
    flt_t maxDisplacement;

    VecList<> tmpGradient;
};