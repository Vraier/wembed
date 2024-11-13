#pragma once

#include "AbstractSimpleEmbedder.hpp"
#include "EmbedderOptions.hpp"
#include "EmbeddedGraph.hpp"
#include "Graph.hpp"
#include "VecList.hpp"

/**
 * Tries to emulate the Force 2 Vec approach.
 * The output is a dot-product embedding
 * Calculates all pair repulsion forces -> O(n^2)
 */
class SimpleF2VEmbedder : public AbstractSimpleEmbedder {
   public:
    SimpleF2VEmbedder(Graph& g, EmbedderOptions opts) : AbstractSimpleEmbedder(g, opts){};

   protected:
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u);
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u);

    virtual void initializeCoordinates();

    double sigmoid(double x);  // sigmoid function
    double dotProduct(CVecRef a, CVecRef b);
};

/**
 * Similar to the F2V approach but normalizes the vectors in every step.
 * The hypothesis is that this will make heterogenous graphs worse, because it the norm of the vector corresponds to the
 * weight.
 */
class SimpleF2VNormedEmbedder : public AbstractSimpleEmbedder {
   public:
    SimpleF2VNormedEmbedder(Graph& g, EmbedderOptions opts) : AbstractSimpleEmbedder(g, opts){};

   protected:
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u);
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u);

    virtual void calculateForceStep();
    virtual void initializeCoordinates();

    double sigmoid(double x);  // sigmoid function
    double dotProduct(CVecRef a, CVecRef b);
};

/**
 * Slight variation of the Fruchtermann embedder. (Repelling forces are only calculated between non neighboring nodes)
 * Is not concerned with weights.
 */
class SimpleFruchtermannEmbedder : public AbstractSimpleEmbedder {
   public:
    SimpleFruchtermannEmbedder(Graph& g, EmbedderOptions opts) : AbstractSimpleEmbedder(g, opts){};

   protected:
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u);
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u);
};

/**
 * Emulates the maxent embedder forces from the paper.
 * It does not use the same loss function minimization.
 */
class SimpleMaxentEmbedder : public AbstractSimpleEmbedder {
   public:
    SimpleMaxentEmbedder(Graph& g, EmbedderOptions opts) : AbstractSimpleEmbedder(g, opts){};

   protected:
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u);
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u);
};

/**
 * Similar to the force to vec approach but uses the euclidean distance as similarity measure.
 *
 * Uses sigmoid edge length and scale.
 */
class SimpleSigmoidEuclideanEmbedder : public AbstractSimpleEmbedder {
   public:
    SimpleSigmoidEuclideanEmbedder(Graph& g, EmbedderOptions opts) : AbstractSimpleEmbedder(g, opts){};

   protected:
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u);
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u);

    double sigmoid(double x);  // sigmoid function
};

/**
 * Similar to the force to vec approach but uses girg embedding space. Does not care about dimensions.
 */
class SimpleSigmoidNoDimEmbedder : public AbstractSimpleEmbedder {
   public:
    SimpleSigmoidNoDimEmbedder(Graph& g, EmbedderOptions opts) : AbstractSimpleEmbedder(g, opts){};

   protected:
    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u);
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u);

    double sigmoid(double x);                      // sigmoid function
    double idealEdgeLength(double wu, double wv);  // ideal edge length for two nodes
};

/**
 * Similar to the force to vec approach but uses the girg embedding space.
 *
 * Also supports weight forces.
 */
class SimpleSigmoidWeightedEmbedder : public AbstractSimpleEmbedder {
   public:
    SimpleSigmoidWeightedEmbedder(Graph& g, EmbedderOptions opts) : AbstractSimpleEmbedder(g, opts){};

    virtual void calculateStep() override;

   protected:
    virtual void calculateWeightStep() override;

    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u) override;
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u) override;

    virtual double weightRepulsionForce(NodeId v, NodeId u) override;
    virtual double weightAttractionForce(NodeId v, NodeId u) override;

    double sigmoid(double x);                      // sigmoid function
    double idealEdgeLength(double wu, double wv);  // ideal edge length for two nodes
};

/**
 * Euclidean embedder that ignores weights and tries to linearize the sigmoid forces
 */
class SimpleLinearEmbedder : public AbstractSimpleEmbedder {
   public:
    SimpleLinearEmbedder(Graph& g, EmbedderOptions opts) : AbstractSimpleEmbedder(g, opts){};

   protected:
    virtual void dumpDebugAtTermination() override;

    virtual TmpCVec<REP_BUFFER> repulsionForce(int v, int u) override;
    virtual TmpCVec<ATTR_BUFFER> attractionForce(int v, int u) override;

    double getSimilarity(double norm, double wu, double wv);

    // how many repulsion forces were calculated at each node and iteration
    // maps (nodeId, iteration) -> numRepForceCalculations
    std::map<std::pair<NodeId, int>, int> numRepForceCalculations;
};
