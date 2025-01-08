#include "SimpleOptimizer.hpp"

SimpleOptimizer::SimpleOptimizer(int dimension, int numNodes, double learningRate, double coolingFactor,
                                 double maxDisplacement)
    : dimension(dimension),
      numNodes(numNodes),
      learningRate(learningRate),
      coolingFactor(coolingFactor),
      maxDisplacement(maxDisplacement),
      tmpGradient(dimension),
      t(0) {
    tmpGradient.setSize(numNodes, 0.0);
}

SimpleOptimizer::~SimpleOptimizer() {}

void SimpleOptimizer::update(VecList& parameters, const VecList& gradients) {
    ASSERT(parameters.size() == numNodes, "Number of nodes in parameters does not match numNodes");
    ASSERT(gradients.size() == numNodes, "Number of nodes in gradients does not match numNodes");

    t++;
    double currCooling = Toolkit::myPow(coolingFactor, t);
    for (int v = 0; v < numNodes; v++) {
        // cap the maximum replacement of the node
        tmpGradient[v] = gradients[v];
        tmpGradient[v].cWiseMax(-maxDisplacement);
        tmpGradient[v].cWiseMin(maxDisplacement);

        tmpGradient[v] *= learningRate * currCooling;
    }

    // apply movement based on force
    for (int v = 0; v < numNodes; v++) {
        parameters[v] += tmpGradient[v];
    }
}

void SimpleOptimizer::reset() {
    t = 0;
    tmpGradient.setAll(0.0);
}
