#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "wembed.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(wembed, m) {
    m.doc() = "WEmbed: weighted low-dimensional graph embeddings";

    py::enum_<wembed::SpatialIndex>(m, "SpatialIndex")
        .value("IndexSNN", wembed::IndexSNN)
        .value("IndexSprk", wembed::IndexSprk)
        .export_values();

    py::class_<wembed::Edge>(m, "Edge")
        .def(py::init<wembed::NodeId, wembed::NodeId>(), py::arg("src"), py::arg("dst"))
        .def_readwrite("src", &wembed::Edge::src)
        .def_readwrite("dst", &wembed::Edge::dst)
        .def("__repr__", [](const wembed::Edge& e) {
            return "Edge(" + std::to_string(e.src) + ", " + std::to_string(e.dst) + ")";
        });

    py::class_<wembed::TimingResult>(m, "TimingResult")
        .def_readonly("depth", &wembed::TimingResult::depth)
        .def_readonly("display_name", &wembed::TimingResult::displayName)
        .def_readonly("value", &wembed::TimingResult::value);

    py::class_<wembed::Loss>(m, "Loss")
        .def_readonly("attractive", &wembed::Loss::attractive)
        .def_readonly("repulsive", &wembed::Loss::repulsive)
        .def_readonly("total", &wembed::Loss::total)
        .def("__repr__", [](const wembed::Loss& l) {
            return "Loss(attractive=" + std::to_string(l.attractive) +
                   ", repulsive=" + std::to_string(l.repulsive) +
                   ", total=" + std::to_string(l.total) + ")";
        });

    py::class_<wembed::Options>(m, "Options")
        .def(py::init<>())
        .def_readwrite("embeddingDimension", &wembed::Options::embeddingDimension)
        .def_readwrite("useUnitWeights", &wembed::Options::useUnitWeights)
        .def_readwrite("dimensionHint", &wembed::Options::dimensionHint)
        .def_readwrite("layeredEmbedding", &wembed::Options::layeredEmbedding)
        .def_readwrite("indexType", &wembed::Options::indexType)
        .def_readwrite("attractionScale", &wembed::Options::attractionScale)
        .def_readwrite("repulsionScale", &wembed::Options::repulsionScale)
        .def_readwrite("centreScale", &wembed::Options::centreScale)
        .def_readwrite("edgeLength", &wembed::Options::edgeLength)
        .def_readwrite("expansionStretch", &wembed::Options::expansionStretch)
        .def_readwrite("coolingFactor", &wembed::Options::coolingFactor)
        .def_readwrite("learningRate", &wembed::Options::learningRate)
        .def_readwrite("maxIterations", &wembed::Options::maxIterations)
        .def_readwrite("positionMinChange", &wembed::Options::positionMinChange);

    py::class_<wembed::Graph>(m, "Graph")
        .def("getNumVertices", &wembed::Graph::getNumVertices)
        .def("getNumEdges", &wembed::Graph::getNumEdges)
        .def("getEdges", &wembed::Graph::getEdges)
        .def("getNeighbors", &wembed::Graph::getNeighbors)
        .def("getNumNeighbors", &wembed::Graph::getNumNeighbors)
        .def("getEdgeTarget", &wembed::Graph::getEdgeTarget)
        .def("areNeighbors", &wembed::Graph::areNeighbors)
        .def("getEdgeList", &wembed::Graph::getEdgeList)
        .def("__repr__", &wembed::Graph::toString);

    py::class_<wembed::Embedder>(m, "Embedder")
        .def("calculateStep", &wembed::Embedder::calculateStep)
        .def("isFinished", &wembed::Embedder::isFinished)
        .def("calculateEmbedding", &wembed::Embedder::calculateEmbedding)
        .def("getNumVertices", &wembed::Embedder::getNumVertices)
        .def("getEmbeddingDimension", &wembed::Embedder::getEmbeddingDimension)
        .def("getCurrentGraph", &wembed::Embedder::getCurrentGraph)
        .def("getCoordinates", &wembed::Embedder::getCoordinates)
        .def("getWeights", &wembed::Embedder::getWeights)
        .def("setCoordinates", &wembed::Embedder::setCoordinates)
        .def("setWeights", &wembed::Embedder::setWeights)
        .def("getTimings", &wembed::Embedder::getTimings)
        .def("getLoss", &wembed::Embedder::getLoss)
        .def("writeCoordinates", &wembed::Embedder::writeCoordinates,
             py::arg("filePath"), py::arg("writeWeights") = true);

    m.def("createEmbedder", &wembed::createEmbedder, py::arg("graph"), py::arg("options"));
    m.def("graphFromEdges", &wembed::graphFromEdges, py::arg("edges"));
    m.def("graphFromEdgeListFile", &wembed::graphFromEdgeListFile,
          py::arg("filePath"), py::arg("comment") = "#", py::arg("delimiter") = " ");
    m.def("readCoordinatesFromFile", &wembed::readCoordinatesFromFile,
          py::arg("filePath"), py::arg("comment") = "%", py::arg("delimiter") = ",");
    m.def("timingsToString", &wembed::timingsToString, py::arg("timings"));
    m.def("setSeed", &wembed::setSeed, py::arg("seed"));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
