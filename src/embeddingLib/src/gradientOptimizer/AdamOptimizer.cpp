#include "AdamOptimizer.hpp"

AdamOptimizer::AdamOptimizer(int dimension, int numNodes, double learningRate, double coolingFactor, double beta1,
                             double beta2, double epsilon)
    : dimension(dimension),
      numNodes(numNodes),
      learningRate(learningRate),
      coolingFactor(coolingFactor),
      beta1(beta1),
      beta2(beta2),
      epsilon(epsilon),
      m(dimension, numNodes),
      v(dimension, numNodes),
      t(0) {}

AdamOptimizer::~AdamOptimizer() {}

void AdamOptimizer::update(VecList& parameters, const VecList& gradients) {
    ASSERT(parameters.size() == numNodes, "Number of nodes in parameters does not match numNodes");
    ASSERT(gradients.size() == numNodes, "Number of nodes in gradients does not match numNodes");

    t++;
    double currCooling = Toolkit::myPow(coolingFactor, t);
#pragma omp parallel for schedule(static)
    for (int n = 0; n < numNodes; n++) {
        for (int i = 0; i < dimension; i++) {
            m[n][i] = beta1 * m[n][i] + (1.0 - beta1) * gradients[n][i];
            v[n][i] = beta2 * v[n][i] + (1.0 - beta2) * gradients[n][i] * gradients[n][i];
            double mHat = m[n][i] / (1.0 - pow(beta1, t));
            double vHat = v[n][i] / (1.0 - pow(beta2, t));
            parameters[n][i] += currCooling * learningRate * mHat / (sqrt(vHat) + epsilon);
        }
    }
}

void AdamOptimizer::update(std::vector<double>& parameters, const std::vector<double>& gradients) {
    ASSERT(dimension == 1, "update with vector needs to be called with dimension 1");
    ASSERT(parameters.size() == numNodes, "Number of nodes in parameters does not match numNodes");
    ASSERT(gradients.size() == numNodes, "Number of nodes in gradients does not match numNodes");

    t++;
    double currCooling = Toolkit::myPow(coolingFactor, t);
//#pragma omp parallel for schedule(static)
    for (int n = 0; n < numNodes; n++) {
        m[n][0] = beta1 * m[n][0] + (1.0 - beta1) * gradients[n];
        v[n][0] = beta2 * v[n][0] + (1.0 - beta2) * gradients[n] * gradients[n];
        double mHat = m[n][0] / (1.0 - pow(beta1, t));
        double vHat = v[n][0] / (1.0 - pow(beta2, t));
        parameters[n] += currCooling * learningRate * mHat / (sqrt(vHat) + epsilon);
    }

}

void AdamOptimizer::reset() {
    m.setAll(0.0);
    v.setAll(0.0);
    t = 0;
}
