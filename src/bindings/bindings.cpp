#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/geometry.hpp>
#include <iostream>

#include "EmbedderOptions.hpp"
#include "EmbeddingIO.hpp"
#include "Graph.hpp"
#include "GraphAlgorithms.hpp"
#include "GraphIO.hpp"
#include "SimpleSamplingEmbedder.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

class PyEmbedderInterface : public EmbedderInterface {
   public:
    using EmbedderInterface::EmbedderInterface;

    void calculateStep() override { PYBIND11_OVERRIDE_PURE(void, EmbedderInterface, calculateStep, ); }

    bool isFinished() override { PYBIND11_OVERRIDE_PURE(bool, EmbedderInterface, isFinished, ); }

    void calculateEmbedding() override { PYBIND11_OVERRIDE_PURE(void, EmbedderInterface, calculateEmbedding, ); }

    std::vector<std::vector<double>> getCoordinates() override {
        PYBIND11_OVERRIDE_PURE(std::vector<std::vector<double>>, EmbedderInterface, getCoordinates, );
    }

    std::vector<double> getWeights() override {
        PYBIND11_OVERRIDE_PURE(std::vector<double>, EmbedderInterface, getWeights, );
    }

    void setCoordinates(const std::vector<std::vector<double>> &coordinates) override {
        PYBIND11_OVERRIDE_PURE(void, EmbedderInterface, setCoordinates, coordinates);
    }

    void setWeights(const std::vector<double> &weights) override {
        PYBIND11_OVERRIDE_PURE(void, EmbedderInterface, setWeights, weights);
    }
};

class PyAbstractSimpleEmbedder : public AbstractSimpleEmbedder {
   public:
    using AbstractSimpleEmbedder::AbstractSimpleEmbedder;

    void calculateStep() override { PYBIND11_OVERRIDE(void, AbstractSimpleEmbedder, calculateStep, ); }

    bool isFinished() override { PYBIND11_OVERRIDE(bool, AbstractSimpleEmbedder, isFinished, ); }

    void calculateEmbedding() override { PYBIND11_OVERRIDE(void, AbstractSimpleEmbedder, calculateEmbedding, ); }

    std::vector<std::vector<double>> getCoordinates() override {
        PYBIND11_OVERRIDE(std::vector<std::vector<double>>, AbstractSimpleEmbedder, getCoordinates, );
    }

    std::vector<double> getWeights() override {
        PYBIND11_OVERRIDE(std::vector<double>, AbstractSimpleEmbedder, getWeights, );
    }

    void setCoordinates(const std::vector<std::vector<double>> &coordinates) override {
        PYBIND11_OVERRIDE(void, AbstractSimpleEmbedder, setCoordinates, coordinates);
    }

    void setWeights(const std::vector<double> &weights) override {
        PYBIND11_OVERRIDE(void, AbstractSimpleEmbedder, setWeights, weights);
    }

   protected:
    TmpCVec<REP_BUFFER> repulsionForce(int v, int u) override {
        PYBIND11_OVERRIDE_PURE(TmpCVec<REP_BUFFER>, AbstractSimpleEmbedder, repulsionForce, v, u);
    }
    TmpCVec<ATTR_BUFFER> attractionForce(int v, int u) override {
        PYBIND11_OVERRIDE_PURE(TmpCVec<ATTR_BUFFER>, AbstractSimpleEmbedder, attractionForce, v, u);
    }
};

class PySimpleSamplingEmbedder : public SimpleSamplingEmbedder {
   public:
    using SimpleSamplingEmbedder::SimpleSamplingEmbedder;

    void calculateStep() override { PYBIND11_OVERRIDE(void, SimpleSamplingEmbedder, calculateStep, ); }

    bool isFinished() override { PYBIND11_OVERRIDE(bool, SimpleSamplingEmbedder, isFinished, ); }

    void calculateEmbedding() override { PYBIND11_OVERRIDE(void, SimpleSamplingEmbedder, calculateEmbedding, ); }

    std::vector<std::vector<double>> getCoordinates() override {
        PYBIND11_OVERRIDE(std::vector<std::vector<double>>, SimpleSamplingEmbedder, getCoordinates, );
    }

    std::vector<double> getWeights() override {
        PYBIND11_OVERRIDE(std::vector<double>, SimpleSamplingEmbedder, getWeights, );
    }

    void setCoordinates(const std::vector<std::vector<double>> &coordinates) override {
        PYBIND11_OVERRIDE(void, SimpleSamplingEmbedder, setCoordinates, coordinates);
    }

    void setWeights(const std::vector<double> &weights) override {
        PYBIND11_OVERRIDE(void, SimpleSamplingEmbedder, setWeights, weights);
    }

   protected:
    TmpCVec<REP_BUFFER> repulsionForce(int v, int u) override {
        PYBIND11_OVERRIDE_PURE(TmpCVec<REP_BUFFER>, SimpleSamplingEmbedder, repulsionForce, v, u);
    }
    TmpCVec<ATTR_BUFFER> attractionForce(int v, int u) override {
        PYBIND11_OVERRIDE_PURE(TmpCVec<ATTR_BUFFER>, SimpleSamplingEmbedder, attractionForce, v, u);
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "WEmbed module for calculating weighted node embeddings";

    // Graphs
    py::class_<Graph>(m, "Graph")
        .def(py::init<std::map<int, std::set<int>> &>(),
             "Construct a graph from a map of sets. The indices have to be > 0 and should be consecutive.")
        .def(py::init<std::vector<std::pair<int, int>> &>(), "Construct a graph from a list of edges.")
        .def("getNumVertices", &Graph::getNumVertices)
        .def("getNumEdges", &Graph::getNumEdges)
        .def("getEdges", &Graph::getEdges)
        .def("getNeighbors", &Graph::getNeighbors)
        .def("getNumNeighbors", &Graph::getNumNeighbors)
        .def("getEdgeContents", &Graph::getEdgeContents)
        .def("getEdgeTarget", &Graph::getEdgeTarget)
        .def("areNeighbors", &Graph::areNeighbors)
        .def("__repr__", [](const Graph &a) { return a.toString(); });

    // Embedder
    py::class_<EmbedderOptions>(m, "EmbedderOptions")
        .def(py::init<>())
        .def_readwrite("dimensionHint", &EmbedderOptions::dimensionHint)
        .def_readwrite("embeddingDimension", &EmbedderOptions::embeddingDimension)
        .def_readwrite("maxIterations", &EmbedderOptions::maxIterations)
        .def_readwrite("speed", &EmbedderOptions::speed)
        .def_readwrite("coolingFactor", &EmbedderOptions::coolingFactor)
        .def("__repr__", [](const EmbedderOptions &a) {
            return "EmbedderOptions(dimensionHint=" + std::to_string(a.dimensionHint) +
                   ", embeddingDimension=" + std::to_string(a.embeddingDimension) +
                   ", maxIterations=" + std::to_string(a.maxIterations) + ", speed=" + std::to_string(a.speed) +
                   ", coolingFactor=" + std::to_string(a.coolingFactor) + ")";
        });

    py::class_<EmbedderInterface, PyEmbedderInterface>(m, "EmbedderInterface")
        .def(py::init<>())
        .def("calculateStep", &EmbedderInterface::calculateStep)
        .def("isFinished", &EmbedderInterface::isFinished)
        .def("calculateEmbedding", &EmbedderInterface::calculateEmbedding)
        .def("getCoordinates", &EmbedderInterface::getCoordinates)
        .def("getWeights", &EmbedderInterface::getWeights)
        .def("setCoordinates", &EmbedderInterface::setCoordinates)
        .def("setWeights", &EmbedderInterface::setWeights);

    py::class_<AbstractSimpleEmbedder, PyAbstractSimpleEmbedder>(m, "AbstractSimpleEmbedder")
        .def(py::init<Graph &, EmbedderOptions>())
        .def("calculateStep", &AbstractSimpleEmbedder::calculateStep)
        .def("isFinished", &AbstractSimpleEmbedder::isFinished)
        .def("calculateEmbedding", &AbstractSimpleEmbedder::calculateEmbedding)
        .def("getCoordinates", &AbstractSimpleEmbedder::getCoordinates)
        .def("getWeights", &AbstractSimpleEmbedder::getWeights)
        .def("setCoordinates", &AbstractSimpleEmbedder::setCoordinates)
        .def("setWeights", &AbstractSimpleEmbedder::setWeights);

    py::class_<SimpleSamplingEmbedder, PySimpleSamplingEmbedder>(m, "Embedder") // TODO: rename when exposing other embedders
        .def(py::init<Graph &, EmbedderOptions>())
        .def("calculateStep", &SimpleSamplingEmbedder::calculateStep)
        .def("isFinished", &SimpleSamplingEmbedder::isFinished)
        .def("calculateEmbedding", &SimpleSamplingEmbedder::calculateEmbedding)
        .def("getCoordinates", &SimpleSamplingEmbedder::getCoordinates)
        .def("getWeights", &SimpleSamplingEmbedder::getWeights)
        .def("setCoordinates", &SimpleSamplingEmbedder::setCoordinates)
        .def("setWeights", &SimpleSamplingEmbedder::setWeights);

    // IO
    m.def("readEdgeList", &GraphIO::readEdgeList);
    m.def("writeCoordinates",
          py::overload_cast<std::string, const std::vector<std::vector<double>> &, const std::vector<double> &>(
              &EmbeddingIO::writeCoordinates));

    // Graph Algorithms
    m.def("isConnected", &GraphAlgo::isConnected);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
