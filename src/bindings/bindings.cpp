#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "EmbeddingIO.hpp"
#include "Graph.hpp"
#include "GraphAlgorithms.hpp"
#include "GraphIO.hpp"
#include "LabelPropagation.hpp"
#include "LayeredEmbedder.hpp"
#include "WEmbedEmbedder.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

class PyEmbedderInterface : public EmbedderInterface {
   public:
    using EmbedderInterface::EmbedderInterface;

    void calculateStep() override { PYBIND11_OVERRIDE_PURE(void, EmbedderInterface, calculateStep); }

    bool isFinished() override { PYBIND11_OVERRIDE_PURE(bool, EmbedderInterface, isFinished); }

    void calculateEmbedding() override { PYBIND11_OVERRIDE_PURE(void, EmbedderInterface, calculateEmbedding); }

    Graph getCurrentGraph() override { PYBIND11_OVERRIDE_PURE(Graph, EmbedderInterface, getCurrentGraph); }

    std::vector<std::vector<double>> getCoordinates() override {
        PYBIND11_OVERRIDE_PURE(std::vector<std::vector<double>>, EmbedderInterface, getCoordinates);
    }

    std::vector<double> getWeights() override {
        PYBIND11_OVERRIDE_PURE(std::vector<double>, EmbedderInterface, getWeights);
    }

    void setCoordinates(const std::vector<std::vector<double>> &coordinates) override {
        PYBIND11_OVERRIDE_PURE(void, EmbedderInterface, setCoordinates, coordinates);
    }

    void setWeights(const std::vector<double> &weights) override {
        PYBIND11_OVERRIDE_PURE(void, EmbedderInterface, setWeights, weights);
    }
};

class PyWEmbedEmbedder : public WEmbedEmbedder {
   public:
    using WEmbedEmbedder::WEmbedEmbedder;

    void calculateStep() override { PYBIND11_OVERRIDE(void, WEmbedEmbedder, calculateStep); }

    bool isFinished() override { PYBIND11_OVERRIDE(bool, WEmbedEmbedder, isFinished); }

    void calculateEmbedding() override { PYBIND11_OVERRIDE(void, WEmbedEmbedder, calculateEmbedding); }

    std::vector<std::vector<double>> getCoordinates() override {
        PYBIND11_OVERRIDE(std::vector<std::vector<double>>, WEmbedEmbedder, getCoordinates);
    }

    std::vector<double> getWeights() override { PYBIND11_OVERRIDE(std::vector<double>, WEmbedEmbedder, getWeights); }

    void setCoordinates(const std::vector<std::vector<double>> &coordinates) override {
        PYBIND11_OVERRIDE(void, WEmbedEmbedder, setCoordinates, coordinates);
    }

    void setWeights(const std::vector<double> &weights) override {
        PYBIND11_OVERRIDE(void, WEmbedEmbedder, setWeights, weights);
    }
};

class PyLayeredEmbedder : public LayeredEmbedder {
   public:
    using LayeredEmbedder::LayeredEmbedder;

    void calculateStep() override { PYBIND11_OVERRIDE(void, LayeredEmbedder, calculateStep); }

    bool isFinished() override { PYBIND11_OVERRIDE(bool, LayeredEmbedder, isFinished); }

    void calculateEmbedding() override { PYBIND11_OVERRIDE(void, LayeredEmbedder, calculateEmbedding); }

    std::vector<std::vector<double>> getCoordinates() override {
        PYBIND11_OVERRIDE(std::vector<std::vector<double>>, LayeredEmbedder, getCoordinates);
    }

    std::vector<double> getWeights() override { PYBIND11_OVERRIDE(std::vector<double>, LayeredEmbedder, getWeights); }

    void setCoordinates(const std::vector<std::vector<double>> &coordinates) override {
        PYBIND11_OVERRIDE(void, LayeredEmbedder, setCoordinates, coordinates);
    }

    void setWeights(const std::vector<double> &weights) override {
        PYBIND11_OVERRIDE(void, LayeredEmbedder, setWeights, weights);
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
        .def_readwrite("cooling", &EmbedderOptions::coolingFactor)
        .def_readwrite("useInfNorm", &EmbedderOptions::useInfNorm)
        .def("__repr__", [](const EmbedderOptions &a) {
            return "EmbedderOptions(dimensionHint=" + std::to_string(a.dimensionHint) +
                   ", embeddingDimension=" + std::to_string(a.embeddingDimension) +
                   ", weightType=" + std::to_string(a.weightType) +
                   ", maxIterations=" + std::to_string(a.maxIterations) + ", speed=" + std::to_string(a.speed) +
                   ", cooling=" + std::to_string(a.coolingFactor) + ", useInfNorm=" + std::to_string(a.useInfNorm) +
                   ")";
        });

    py::class_<EmbedderInterface, PyEmbedderInterface>(m, "EmbedderInterface")
        .def("calculateStep", &EmbedderInterface::calculateStep)
        .def("isFinished", &EmbedderInterface::isFinished)
        .def("calculateEmbedding", &EmbedderInterface::calculateEmbedding)
        .def("getCoordinates", &EmbedderInterface::getCoordinates)
        .def("getWeights", &EmbedderInterface::getWeights)
        .def("setCoordinates", &EmbedderInterface::setCoordinates)
        .def("setWeights", &EmbedderInterface::setWeights);

    py::class_<WEmbedEmbedder, PyWEmbedEmbedder>(m, "Embedder")
        .def(py::init<Graph &, EmbedderOptions>())
        .def("calculateStep", &WEmbedEmbedder::calculateStep)
        .def("isFinished", &WEmbedEmbedder::isFinished)
        .def("calculateEmbedding", &WEmbedEmbedder::calculateEmbedding)
        .def("getCoordinates", &WEmbedEmbedder::getCoordinates)
        .def("getWeights", &WEmbedEmbedder::getWeights)
        .def("setCoordinates", &WEmbedEmbedder::setCoordinates)
        .def("setWeights", &WEmbedEmbedder::setWeights)
        .def("getTimings", &WEmbedEmbedder::getTimings);

    py::class_<LayeredEmbedder, PyLayeredEmbedder>(m, "LayeredEmbedder")
        .def(py::init<Graph &, LabelPropagation &, EmbedderOptions>())
        .def("calculateStep", &LayeredEmbedder::calculateStep)
        .def("isFinished", &LayeredEmbedder::isFinished)
        .def("calculateEmbedding", &LayeredEmbedder::calculateEmbedding)
        .def("getCoordinates", &LayeredEmbedder::getCoordinates)
        .def("getWeights", &LayeredEmbedder::getWeights)
        .def("setCoordinates", &LayeredEmbedder::setCoordinates)
        .def("setWeights", &LayeredEmbedder::setWeights);

    // LablePropagation

    py::class_<PartitionerOptions>(m, "PartitionerOptions").def(py::init<>());

    py::class_<LabelPropagation>(m, "LabelPropagation")
        .def(py::init<PartitionerOptions, Graph &, std::vector<double> &>())
        .def("initialize", &LabelPropagation::coarsenAllLayers);

    // Other functions
    m.def("readEdgeList", &GraphIO::readEdgeList, py::arg("filePath"), py::arg("comment") = "#",
          py::arg("delimiter") = " ");
    m.def("writeCoordinates",
          py::overload_cast<std::string, const std::vector<std::vector<double>> &, const std::vector<double> &>(
              &EmbeddingIO::writeCoordinates));
    m.def("isConnected", &GraphAlgo::isConnected);
    m.def("setSeed", &Rand::setSeed);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
