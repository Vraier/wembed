#pragma once

#include "Optimizer.hpp"

class AdamOptimizer : public Optimizer {
   public:
    AdamOptimizer(int dimension, int numNodes, float beta1, float beta2, float epsilon);
    ~AdamOptimizer();

    void update(VecList<>& parameters, const VecList<>& gradients, float learningRate) override;
    void reset() override;

   private:
    int dimension;
    int numNodes;
    float beta1;
    float beta2;
    float epsilon;

    VecList<> m;  // First moment estimates
    VecList<> v;  // Second moment estimates
    int t;      // Time step, only used for bias correction
};