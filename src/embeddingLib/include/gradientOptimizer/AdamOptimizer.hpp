#pragma once

#include "Optimizer.hpp"

class AdamOptimizer : public Optimizer {
   public:
    AdamOptimizer(int dimension, int numNodes, double beta1, double beta2, double epsilon);
    ~AdamOptimizer();

    void update(VecList& parameters, const VecList& gradients, double learningRate) override;
    void reset() override;

   private:
    int dimension;
    int numNodes;
    double beta1;
    double beta2;
    double epsilon;

    VecList m;  // First moment estimates
    VecList v;  // Second moment estimates
    int t;      // Time step, only used for bias correction
};