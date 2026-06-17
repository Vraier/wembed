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
#include "NewWEmbedEmbedder.hpp"
#include "Timings.hpp"

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

    std::vector<util::TimingResult> getTimings() override {
        PYBIND11_OVERRIDE_PURE(std::vector<util::TimingResult>, EmbedderInterface, getTimings);
    }

    void setCoordinates(const std::vector<std::vector<double>> &coordinates) override {
        PYBIND11_OVERRIDE_PURE(void, EmbedderInterface, setCoordinates, coordinates);
    }

    void setWeights(const std::vector<double> &weights) override {
        PYBIND11_OVERRIDE_PURE(void, EmbedderInterface, setWeights, weights);
    }
};

class PyNewWEmbedEmbedder : public NewWEmbedEmbedder {
   public:
    using NewWEmbedEmbedder::NewWEmbedEmbedder;

    void calculateStep() override { PYBIND11_OVERRIDE(void, NewWEmbedEmbedder, calculateStep); }

    bool isFinished() override { PYBIND11_OVERRIDE(bool, NewWEmbedEmbedder, isFinished); }

    void calculateEmbedding() override { PYBIND11_OVERRIDE(void, NewWEmbedEmbedder, calculateEmbedding); }

    Graph getCurrentGraph() override { PYBIND11_OVERRIDE(Graph, NewWEmbedEmbedder, getCurrentGraph); }

    std::vector<std::vector<double>> getCoordinates() override {
        PYBIND11_OVERRIDE(std::vector<std::vector<double>>, NewWEmbedEmbedder, getCoordinates);
    }

    std::vector<double> getWeights() override { PYBIND11_OVERRIDE(std::vector<double>, NewWEmbedEmbedder, getWeights); }

    std::vector<util::TimingResult> getTimings() override {
        PYBIND11_OVERRIDE(std::vector<util::TimingResult>, NewWEmbedEmbedder, getTimings);
    }

    void setCoordinates(const std::vector<std::vector<double>> &coordinates) override {
        PYBIND11_OVERRIDE(void, NewWEmbedEmbedder, setCoordinates, coordinates);
    }

    void setWeights(const std::vector<double> &weights) override {
        PYBIND11_OVERRIDE(void, NewWEmbedEmbedder, setWeights, weights);
    }
};

class PyLayeredEmbedder : public LayeredEmbedder {
   public:
    using LayeredEmbedder::LayeredEmbedder;

    void calculateStep() override { PYBIND11_OVERRIDE(void, LayeredEmbedder, calculateStep); }

    bool isFinished() override { PYBIND11_OVERRIDE(bool, LayeredEmbedder, isFinished); }

    void calculateEmbedding() override { PYBIND11_OVERRIDE(void, LayeredEmbedder, calculateEmbedding); }

    Graph getCurrentGraph() override { PYBIND11_OVERRIDE(Graph, LayeredEmbedder, getCurrentGraph); }

    std::vector<std::vector<double>> getCoordinates() override {
        PYBIND11_OVERRIDE(std::vector<std::vector<double>>, LayeredEmbedder, getCoordinates);
    }

    std::vector<double> getWeights() override { PYBIND11_OVERRIDE(std::vector<double>, LayeredEmbedder, getWeights); }

    std::vector<util::TimingResult> getTimings() override {
        PYBIND11_OVERRIDE(std::vector<util::TimingResult>, LayeredEmbedder, getTimings);
    }

    void setCoordinates(const std::vector<std::vector<double>> &coordinates) override {
        PYBIND11_OVERRIDE(void, LayeredEmbedder, setCoordinates, coordinates);
    }

    void setWeights(const std::vector<double> &weights) override {
        PYBIND11_OVERRIDE(void, LayeredEmbedder, setWeights, weights);
    }
};

PYBIND11_MODULE(wembed, m) {
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

    // Timings
    py::class_<util::TimingResult>(m, "TimingResult")
        .def_readonly("depth", &util::TimingResult::depth)
        .def_readonly("display_name", &util::TimingResult::display_name)
        .def_readonly("value", &util::TimingResult::value)
        .def("__repr__", [](const util::TimingResult &t) {
            return "TimingResult(depth=" + std::to_string(t.depth) +
                   ", display_name=" + t.display_name +
                   ", value=" + std::to_string(t.value) + ")";
        });

    // Embedder options
    py::class_<EmbedderOptions>(m, "EmbedderOptions")
        .def(py::init<>())
        .def_readwrite("dimensionHint", &EmbedderOptions::dimensionHint)
        .def_readwrite("embeddingDimension", &EmbedderOptions::embeddingDimension)
        .def_readwrite("maxIterations", &EmbedderOptions::maxIterations)
        .def_readwrite("speed", &EmbedderOptions::learningRate)
        .def_readwrite("cooling", &EmbedderOptions::coolingFactor)
        .def("__repr__", [](const EmbedderOptions &a) {
            return "EmbedderOptions(dimensionHint=" + std::to_string(a.dimensionHint) +
                   ", embeddingDimension=" + std::to_string(a.embeddingDimension) +
                   ", weightType=" + std::to_string(static_cast<int>(a.weightType)) +
                   ", maxIterations=" + std::to_string(a.maxIterations) +
                   ", speed=" + std::to_string(a.learningRate) +
                   ", cooling=" + std::to_string(a.coolingFactor) + ")";
        });

    py::class_<EmbedderInterface, PyEmbedderInterface>(m, "EmbedderInterface")
        .def("calculateStep", &EmbedderInterface::calculateStep)
        .def("isFinished", &EmbedderInterface::isFinished)
        .def("calculateEmbedding", &EmbedderInterface::calculateEmbedding)
        .def("getCoordinates", &EmbedderInterface::getCoordinates)
        .def("getWeights", &EmbedderInterface::getWeights)
        .def("getTimings", &EmbedderInterface::getTimings)
        .def("setCoordinates", &EmbedderInterface::setCoordinates)
        .def("setWeights", &EmbedderInterface::setWeights);

    py::class_<NewWEmbedEmbedder, EmbedderInterface, PyNewWEmbedEmbedder>(m, "Embedder")
        .def(py::init<const Graph &, const EmbedderOptions &>())
        .def("calculateStep", &NewWEmbedEmbedder::calculateStep)
        .def("isFinished", &NewWEmbedEmbedder::isFinished)
        .def("calculateEmbedding", &NewWEmbedEmbedder::calculateEmbedding)
        .def("getCoordinates", &NewWEmbedEmbedder::getCoordinates)
        .def("getWeights", &NewWEmbedEmbedder::getWeights)
        .def("getTimings", &NewWEmbedEmbedder::getTimings)
        .def("setCoordinates", &NewWEmbedEmbedder::setCoordinates)
        .def("setWeights", &NewWEmbedEmbedder::setWeights);

    py::class_<LayeredEmbedder, EmbedderInterface, PyLayeredEmbedder>(m, "LayeredEmbedder")
        .def(py::init<Graph &, LabelPropagation &, EmbedderOptions>())
        .def("calculateStep", &LayeredEmbedder::calculateStep)
        .def("isFinished", &LayeredEmbedder::isFinished)
        .def("calculateEmbedding", &LayeredEmbedder::calculateEmbedding)
        .def("getCoordinates", &LayeredEmbedder::getCoordinates)
        .def("getWeights", &LayeredEmbedder::getWeights)
        .def("getTimings", &LayeredEmbedder::getTimings)
        .def("setCoordinates", &LayeredEmbedder::setCoordinates)
        .def("setWeights", &LayeredEmbedder::setWeights);

    // LabelPropagation
    py::class_<PartitionerOptions>(m, "PartitionerOptions").def(py::init<>());

    py::class_<LabelPropagation>(m, "LabelPropagation")
        .def(py::init<PartitionerOptions, Graph &, std::vector<double> &>())
        .def("initialize", &LabelPropagation::coarsenAllLayers);

    // Free functions
    m.def("timingsToString", &util::timingsToStringRepresentation, py::arg("timings"));

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
