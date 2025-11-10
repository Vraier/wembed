#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "EmbeddingIO.hpp"
#include "Graph.hpp"
#include "GraphIO.hpp"
#include "LabelPropagation.hpp"
#include "LayeredEmbedder.hpp"
#include "Partitioner.hpp"
#include "Rand.hpp"
#include "WEmbedEmbedder.hpp"

// note: this is a really hacky way to solve name conflicts
namespace impl {
    using EmbeddingGraph = Graph;
    using EmbedderInterface = EmbedderInterface;
}

#define _WEMBED_IS_IMPL
#include "wembed.h"


namespace wembed {

Graph::Graph(std::unique_ptr<impl::EmbeddingGraph>&& graph):
    _graph(std::move(graph)) { }

// must be in cpp file because unique_ptr requires that the definition is available
Graph::~Graph() = default;
Graph::Graph(Graph&& other) = default;
Graph& Graph::operator=(Graph&& other) = default;

// global information
NodeId Graph::getNumVertices() const {
    return _graph->getNumVertices();
}

EdgeId Graph::getNumEdges() const {
    return _graph->getNumEdges();
}

// // neighborhood information
std::vector<EdgeId> Graph::getEdges(NodeId v) const {
    return _graph->getEdges(v);
}

std::vector<NodeId> Graph::getNeighbors(NodeId v) const {
    return _graph->getNeighbors(v);
}

int Graph::getNumNeighbors(NodeId v) const {
    return _graph->getNumNeighbors(v);
}

NodeId Graph::getEdgeTarget(EdgeId e) const {
    return _graph->getEdgeTarget(e);
}

bool Graph::areNeighbors(NodeId v, NodeId u) const {
    return _graph->areNeighbors(v, u);
}

std::string Graph::toString() const {
    return _graph->toString();
}


Embedder::Embedder(std::unique_ptr<impl::EmbedderInterface>&& embedder):
    _embedder(std::move(embedder)) { }

// must be in cpp file because unique_ptr requires that the definition is available
Embedder::~Embedder() = default;
Embedder::Embedder(Embedder&& other) = default;
Embedder& Embedder::operator=(Embedder&& other) = default;

void Embedder::calculateStep() {
    _embedder->calculateStep();
}

bool Embedder::isFinished() {
    return _embedder->isFinished();
}

void Embedder::calculateEmbedding() {
    _embedder->calculateEmbedding();
}

Graph Embedder::getCurrentGraph() const {
    auto graph = _embedder->getCurrentGraph();
    return Graph(std::make_unique<impl::EmbeddingGraph>(std::move(graph)));
}

std::vector<std::vector<double>> Embedder::getCoordinates() const {
    return _embedder->getCoordinates();
}

std::vector<double> Embedder::getWeights() const {
    return _embedder->getWeights();
}

void Embedder::setCoordinates(const std::vector<std::vector<double>>& coordinates) {
    _embedder->setCoordinates(coordinates);
}

void Embedder::setWeights(const std::vector<double>& weights) {
    _embedder->setWeights(weights);
}

void Embedder::writeCoordinates(const std::string& filePath, bool writeWeights) const {
    if (writeWeights) {
        EmbeddingIO::writeCoordinates(filePath, getCoordinates(), getWeights());
    } else {
        EmbeddingIO::writeCoordinates(filePath, getCoordinates());
    }
}


Embedder createEmbedder(const Graph& g, const Options& options) {
    EmbedderOptions opts;
    opts.embeddingDimension = options.embeddingDimension;
    opts.weightType = options.useWeights ? WeightType::Degree : WeightType::Unit;
    opts.dimensionHint = options.dimensionHint;
    opts.coolingFactor = options.coolingFactor;
    opts.learningRate = options.learningRate;
    opts.maxIterations = options.maxIterations;
    opts.indexType = options.indexType == SpatialIndex::RTree ? IndexType::RTree : IndexType::SNN;

    const auto& graph = *g._graph;
    if (options.layeredEmbedding) {
        LabelPropagation coarsener(PartitionerOptions{}, graph,
                                   std::vector<double>(g.getNumEdges() * 2, 1.0));
        return Embedder(std::make_unique<LayeredEmbedder>(graph, coarsener, opts));
    } else {
        return Embedder(std::make_unique<WEmbedEmbedder>(graph, opts));
    }
}

Graph graphFromEdges(const std::vector<std::pair<NodeId, NodeId>>& edges) {
    return Graph(std::make_unique<impl::EmbeddingGraph>(edges));
}

Graph graphFromNeighborhoods(const std::map<NodeId, std::set<NodeId>>& nodeToNeighborhood) {
    return Graph(std::make_unique<impl::EmbeddingGraph>(nodeToNeighborhood));
}

Graph graphFromEdgeListFile(const std::string& filePath,
                            const std::string& comment,
                            const std::string& delimiter) {
    auto graph = GraphIO::readEdgeList(filePath, comment, delimiter);
    return Graph(std::make_unique<impl::EmbeddingGraph>(std::move(graph)));
}

void setSeed(int seed) {
    Rand::setSeed(seed);
}

}  // namespace wembed
