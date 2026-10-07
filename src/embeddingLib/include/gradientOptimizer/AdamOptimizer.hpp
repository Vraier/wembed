#pragma once

#include "Optimizer.hpp"

class AdamOptimizer : public Optimizer {
   public:
    AdamOptimizer(int dimension, int numNodes, flt_t beta1, flt_t beta2, flt_t epsilon);
    ~AdamOptimizer();

    void update(VecList<>& parameters, const VecList<>& gradients, flt_t learningRate) override;
    void reset() override;

   private:
    int dimension;
    int numNodes;
    flt_t beta1;
    flt_t beta2;
    flt_t epsilon;

    VecList<> m;  // First moment estimates
    VecList<> v;  // Second moment estimates
    int t;      // Time step, only used for bias correction
};