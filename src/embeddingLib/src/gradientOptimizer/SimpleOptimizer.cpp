#include "SimpleOptimizer.hpp"

SimpleOptimizer::SimpleOptimizer(int dimension, int numNodes, double maxDisplacement)
    : dimension(dimension),
      numNodes(numNodes),
      maxDisplacement(maxDisplacement),
      tmpGradient(dimension) {
    tmpGradient.setSize(numNodes, 0.0);
}

SimpleOptimizer::~SimpleOptimizer() {}

void SimpleOptimizer::update(VecList& parameters, const VecList& gradients, double learningRate) {
    ASSERT(parameters.size() == numNodes, "Number of nodes in parameters does not match numNodes");
    ASSERT(gradients.size() == numNodes, "Number of nodes in gradients does not match numNodes");

    for (int v = 0; v < numNodes; v++) {
        // cap the maximum replacement of the node
        tmpGradient[v] = gradients[v];
        tmpGradient[v].cWiseMax(-maxDisplacement);
        tmpGradient[v].cWiseMin(maxDisplacement);

        tmpGradient[v] *= learningRate;
    }

    // apply movement based on force
    for (int v = 0; v < numNodes; v++) {
        parameters[v] += tmpGradient[v];
    }
}

void SimpleOptimizer::reset() {
    tmpGradient.setAll(0.0);
}
