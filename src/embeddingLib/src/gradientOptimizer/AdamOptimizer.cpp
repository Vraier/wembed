#include "AdamOptimizer.hpp"

AdamOptimizer::AdamOptimizer(int dimension, int numNodes, flt_t beta1, flt_t beta2, flt_t epsilon)
    : dimension(dimension),
      numNodes(numNodes),
      beta1(beta1),
      beta2(beta2),
      epsilon(epsilon),
      m(dimension, numNodes),
      v(dimension, numNodes),
      t(0) {}

AdamOptimizer::~AdamOptimizer() {}

void AdamOptimizer::update(VecList<>& parameters, const VecList<>& gradients, flt_t learningRate) {
    ASSERT(parameters.size() == numNodes, "Number of nodes in parameters does not match numNodes");
    ASSERT(gradients.size() == numNodes, "Number of nodes in gradients does not match numNodes");

    t++;
#pragma omp parallel for schedule(static)
    for (int n = 0; n < numNodes; n++) {
        for (int i = 0; i < dimension; i++) {
            m[n][i] = beta1 * m[n][i] + (flt_t{1.0} - beta1) * gradients[n][i];
            v[n][i] = beta2 * v[n][i] + (flt_t{1.0} - beta2) * gradients[n][i] * gradients[n][i];
            flt_t mHat = m[n][i] / (flt_t{1.0} - std::pow(beta1, t));
            flt_t vHat = v[n][i] / (flt_t{1.0} - std::pow(beta2, t));
            parameters[n][i] += learningRate * mHat / (std::sqrt(vHat) + epsilon);
        }
    }
}

void AdamOptimizer::reset() {
    m.setAll(0.0);
    v.setAll(0.0);
    t = 0;
}
