#pragma once

#include "Optimizer.hpp"

class AdamOptimizer : public Optimizer {
   public:
    AdamOptimizer(int dimension, int numNodes, double learningRate, double coolingFactor, double beta1, double beta2, double epsilon);
    ~AdamOptimizer();

    void update(VecList& parameters, const VecList& gradients) override;
    void reset() override;

   private:
    int dimension;
    int numNodes;
    double learningRate;
    double coolingFactor;
    double beta1;
    double beta2;
    double epsilon;

    VecList m;  // First moment estimates
    VecList v;  // Second moment estimates
    int t;      // Time step
};