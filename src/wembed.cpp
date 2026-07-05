#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "EmbeddingIO.hpp"
#include "Graph.hpp"
#include "GraphIO.hpp"
#include "LabelPropagation.hpp"
#include "LayeredEmbedder.hpp"
#include "NewWEmbedEmbedder.hpp"
#include "Partitioner.hpp"
#include "Rand.hpp"
#include "Timings.hpp"

namespace impl {
    using EmbeddingGraph = ::Graph;
    using EmbedderInterface = ::EmbedderInterface;
}

#define _WEMBED_IS_IMPL
#include "wembed.h"


namespace wembed {

Graph::Graph(std::unique_ptr<impl::EmbeddingGraph>&& graph):
    _graph(std::move(graph)) { }

Graph::~Graph() = default;
Graph::Graph(Graph&& other) = default;
Graph& Graph::operator=(Graph&& other) = default;

NodeId Graph::getNumVertices() const {
    return _graph->getNumVertices();
}

EdgeId Graph::getNumEdges() const {
    return _graph->getNumEdges();
}

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

std::vector<Edge> Graph::getEdgeList() const {
    std::vector<Edge> out;
    out.reserve(_graph->getNumEdges());
    const NodeId n = _graph->getNumVertices();
    for (NodeId v = 0; v < n; ++v) {
        for (NodeId u : _graph->getNeighbors(v)) {
            if (v < u) {
                out.push_back({v, u});
            }
        }
    }
    return out;
}

std::string Graph::toString() const {
    return _graph->toString();
}


Embedder::Embedder(std::unique_ptr<impl::EmbedderInterface>&& embedder):
    _embedder(std::move(embedder)) { }

Embedder::~Embedder() = default;
Embedder::Embedder(Embedder&& other) = default;
Embedder& Embedder::operator=(Embedder&& other) = default;

void Embedder::calculateStep() {
    _embedder->calculateStep();
}

bool Embedder::isFinished() const {
    return _embedder->isFinished();
}

void Embedder::calculateEmbedding() {
    _embedder->calculateEmbedding();
}

int32_t Embedder::getNumVertices() const {
    return static_cast<int32_t>(_embedder->getNumVertices());
}

int32_t Embedder::getEmbeddingDimension() const {
    return static_cast<int32_t>(_embedder->getEmbeddingDimension());
}

void Embedder::copyCoordinatesTo(double* out) const {
    _embedder->copyCoordinatesTo(out);
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

std::vector<TimingResult> Embedder::getTimings() const {
    auto internal = _embedder->getTimings();
    std::vector<TimingResult> out;
    out.reserve(internal.size());
    for (const auto& t : internal) {
        out.push_back({static_cast<uint64_t>(t.depth), t.display_name, t.value});
    }
    return out;
}

void Embedder::writeCoordinates(const std::string& filePath, bool writeWeights) const {
    if (writeWeights) {
        EmbeddingIO::writeCoordinates(filePath, _embedder->getCoordinates(), _embedder->getWeights());
    } else {
        EmbeddingIO::writeCoordinates(filePath, _embedder->getCoordinates());
    }
}


static IndexType toInternalIndexType(SpatialIndex idx) {
    switch (idx) {
        case IndexSNN:  return IndexType::SNN;
        case IndexSprk: return IndexType::Sprk;
    }
    return IndexType::Sprk;
}

Embedder createEmbedder(const Graph& g, const Options& options) {
    EmbedderOptions opts;
    opts.embeddingDimension = options.embeddingDimension;
    opts.weightType = options.useUnitWeights ? WeightType::Unit : WeightType::Degree;
    opts.dimensionHint = options.dimensionHint;
    opts.indexType = toInternalIndexType(options.indexType);
    opts.attractionScale = options.attractionScale;
    opts.repulsionScale = options.repulsionScale;
    opts.edgeLength = options.edgeLength;
    opts.expansionStretch = options.expansionStretch;
    opts.coolingFactor = options.coolingFactor;
    opts.learningRate = options.learningRate;
    opts.maxIterations = options.maxIterations;
    opts.positionMinChange = options.positionMinChange;

    // LayeredEmbedder/LabelPropagation constructors take a non-const Graph&,
    // but neither actually mutates it; cast away const for compatibility.
    auto& graph = const_cast<impl::EmbeddingGraph&>(*g._graph);
    if (options.layeredEmbedding) {
        std::vector<double> edgeWeights(g.getNumEdges() * 2, 1.0);
        auto coarsener = std::make_unique<LabelPropagation>(PartitionerOptions{}, graph, edgeWeights);
        return Embedder(std::make_unique<LayeredEmbedder>(graph, *coarsener, opts));
    } else {
        return Embedder(std::make_unique<NewWEmbedEmbedder>(graph, opts));
    }
}

Graph graphFromEdges(const std::vector<Edge>& edges) {
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(edges.size());
    for (const auto& e : edges) {
        pairs.emplace_back(e.src, e.dst);
    }
    return Graph(std::make_unique<impl::EmbeddingGraph>(pairs));
}

Graph graphFromEdgeListFile(const std::string& filePath,
                            const std::string& comment,
                            const std::string& delimiter) {
    auto graph = GraphIO::readEdgeList(filePath, comment, delimiter);
    return Graph(std::make_unique<impl::EmbeddingGraph>(std::move(graph)));
}

std::vector<std::vector<double>> readCoordinatesFromFile(const std::string& filePath,
                                                          const std::string& comment,
                                                          const std::string& delimiter) {
    return EmbeddingIO::readCoordinatesFromFile(filePath, comment, delimiter);
}

std::string timingsToString(const std::vector<TimingResult>& timings) {
    std::vector<util::TimingResult> internal;
    internal.reserve(timings.size());
    for (const auto& t : timings) {
        internal.push_back({static_cast<size_t>(t.depth), t.displayName, t.value});
    }
    return util::timingsToStringRepresentation(internal);
}

void setSeed(int seed) {
    Rand::setSeed(seed);
}

}  // namespace wembed
