#include "HierarchyEmbedder.hpp"

#include "EmbedderFactory.hpp"
#include "FruchtermannLayerEmbedder.hpp"
#include "IPEDrawer.hpp"
#include "LabelPropagation.hpp"
#include "Partitioner.hpp"
#include "SFMLDrawer.hpp"
#include "SVGDrawer.hpp"
#include "SigmoidLayerEmbedder.hpp"
#include "WeightedApproxEmbedder.hpp"
#include "WeightedSpringEmbedder.hpp"
#include "WeightedStressEmbedder.hpp"

void HierarchyEmbedder::setInitialWeights(const std::vector<double>& weights) { initialWeights = weights; }

void HierarchyEmbedder::initializeNewRun() {
    const int N = originalGraph.getNumVertices();

    switch (opts.embedderOptions.weightType) {
        case 0:  // unit weight
            initialWeights = std::vector<double>(N, 1.0);
            break;
        case 1:  // set weight to degree
            initialWeights = std::vector<double>(N);

            for (NodeId v = 0; v < N; v++) {
                initialWeights[v] = originalGraph.getNumNeighbors(v);
            }
            break;
        case 2:  // noting to do, assume that weights have been set previously
            if (initialWeights.size() != N) {
                LOG_ERROR( "Initial weights have not been set correctly");
            }
            break;
        default:
            LOG_ERROR( "Unknown inital weights choosen");
            break;
    }

    // scale the weights to fit to the assumed dimension of them input graph
    if (opts.embedderOptions.dimensionHint > 0) {
        for (NodeId v = 0; v < N; v++) {
            initialWeights[v] =
                Toolkit::myPow(initialWeights[v], (double)opts.dimension / (double)opts.embedderOptions.dimensionHint);
        }
    }
    // scale the weights so that the mean is 1
    double weightSum = 0;
    for (NodeId v = 0; v < N; v++) {
        weightSum += initialWeights[v];
    }
    for (NodeId v = 0; v < N; v++) {
        initialWeights[v] *= ((double)N / weightSum);
    }

    graphHierarchy = buildHierarchy();
}

void HierarchyEmbedder::calculateEmbedding() {
    SVGOutputWriter writer;
    WeightedApproxEmbedder* embedder;
    SFMLDrawer* drawer = nullptr;

    if (opts.animate) {
        drawer = new SFMLDrawer();
    }

    for (int level = graphHierarchy->getNumLayers() - 2; level >= 0; level--) {
        // expand the nodes and update the tree
        expandPositions(level);

        int currN = graphHierarchy->getLayerSize(level);
        LOG_INFO( "Embedding Hierarchy " << level << " with " << currN << " nodes");

        embedder = getEmbedderForLayer(level);
        embedder->initializeNewRun();

        std::vector<double> weights(graphHierarchy->graphs[level].getNumVertices());
        for (NodeId v = 0; v < graphHierarchy->graphs[level].getNumVertices(); v++) {
            weights[v] = graphHierarchy->graphs[level].getNodeWeight(v);
        }
        std::vector<double> parentIds;
        for (NodeId v = 0; v < graphHierarchy->getLayerSize(level); v++) {
            int currLevel = level;
            int currParent = v;
            while (currLevel < graphHierarchy->getNumLayers() - 2) {
                currParent = graphHierarchy->getParent(currLevel, currParent);
                currLevel++;
            }
            parentIds.push_back(currParent);
        }

        int numSteps = 0;
        while (!embedder->isFinished()) {
            numSteps++;
            embedder->calculateStep();
            if (opts.animate) {
                drawer->processFrame(
                    graphHierarchy->graphs[level].getGraph(),
                    Common::projectOntoPlane(graphHierarchy->graphs[level].coordinates.convertToVector()), parentIds);
            }
        }

        LOG_INFO( "Finished Embedding after " << numSteps << " steps");

        delete embedder;
        if (opts.writerOptions.svgPath != "") {
            writer.write(opts.writerOptions.svgPath + "_color.svg", graphHierarchy->graphs[level].getGraph(),
                         graphHierarchy->graphs[level].coordinates.convertToVector());
            // parentIds);
        }
        if (opts.writerOptions.ipePath != "") {
            IpeOutputWriter ipeWriter(opts.writerOptions.ipePath + std::to_string(level) + ".ipe", 3, 50);
            ipeWriter.write_graph(graphHierarchy->graphs[level].getGraph(),
                                  graphHierarchy->graphs[level].coordinates.convertToVector(), parentIds);
        }
    }

    if (opts.animate) {
        delete drawer;
    }
    LOG_INFO( "Finished all Hierarchies");
}

EmbeddedGraph HierarchyEmbedder::getEmbeddedGraph() { return graphHierarchy->graphs[0]; }

GraphHierarchy* HierarchyEmbedder::buildHierarchy() {
    // calculate inital weights for partitioner
    std::vector<double> edgeWeights(originalGraph.getNumEdges() * 2);
    for (NodeId v = 0; v < originalGraph.getNumVertices(); v++) {
        for (EdgeId e : originalGraph.getEdges(v)) {
            edgeWeights[e] = 1.0 / Toolkit::myPow(initialWeights[v] * initialWeights[originalGraph.getEdgeTarget(e)],
                                            1.0 / (double)opts.dimension);
        }
    }

    Partitioner* partitioner;
    switch (opts.partitionerOptions.partitionType) {
        case 0:
            partitioner = new LabelPropagation(opts.partitionerOptions, originalGraph, edgeWeights);
            break;
        default:
            LOG_ERROR( "Unknown partition type");
            return nullptr;
    }
    ParentPointerTree parents = partitioner->coarsenAllLayers();
    delete partitioner;
    return new GraphHierarchy({opts.dimension, opts.embedderOptions.forceExponent}, originalGraph, parents,
                              initialWeights);
}

void HierarchyEmbedder::expandPositions(int currLayer) {
    // LOG_INFO( "Expanding coordinates");

    VecList hackVec(opts.dimension);
    hackVec.setSize(1, 0);

    if (currLayer == graphHierarchy->getNumLayers() - 1) {
        for (NodeId v = 0; v < graphHierarchy->getLayerSize(currLayer); v++) {
            graphHierarchy->setPositionOfNode(currLayer, v, hackVec[0]);
        }
        return;
    }

    // set the current coordinates to the coordinates of the layer above
    graphHierarchy->applyPositionToChildren(currLayer + 1);
    int currN = graphHierarchy->getLayerSize(currLayer);

    for (int v = 0; v < currN; v++) {
        int parent = graphHierarchy->getParent(currLayer, v);

        double stretch =
            Toolkit::myPow(graphHierarchy->getTotalContainedNodes(currLayer + 1, parent), 1.0 / (double)opts.dimension);
        stretch = 1.0;

        hackVec[0].setToRandomUnitVector();
        hackVec[0] *= stretch;
        hackVec[0] += graphHierarchy->getAveragePosition(currLayer, v);
        graphHierarchy->setPositionOfNode(currLayer, v, hackVec[0]);
    }
    // LOG_INFO( "Expanded coordinates");
}

WeightedApproxEmbedder* HierarchyEmbedder::getEmbedderForLayer(int currLayer) {
    return EmbedderFactory::constructLayerEmbedder(opts, graphHierarchy, currLayer);
}
