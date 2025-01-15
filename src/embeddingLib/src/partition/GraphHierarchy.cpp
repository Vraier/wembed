#include "GraphHierarchy.hpp"

#include "GraphAlgorithms.hpp"

GraphHierarchy::GraphHierarchy(const Graph& graph, LabelPropagation& coarsener) {
    LOG_INFO("Starting to build graph hierarchy");

    ParentPointerTree parentPointer = coarsener.coarsenAllLayers();
    NUMLAYERS = parentPointer.size();

    // coarsen the graphs and convert them to embedded graphs
    LOG_INFO("Coarsening graphs");
    std::vector<std::vector<EdgeId>> edgeParentPointers;
    Graph currGraph = graph;
    for (int i = 0; i < NUMLAYERS; i++) {
        LOG_INFO("... in layer " << i);
        graphs.push_back(currGraph);
        if (i < NUMLAYERS - 1) {
            auto coarsened = GraphAlgo::coarsenGraph(currGraph, parentPointer[i]);
            currGraph = coarsened.first;
            edgeParentPointers.push_back(coarsened.second);
        }
    }
    // push back an empty pointer. last graph only has a single node and no edges
    edgeParentPointers.push_back(std::vector<EdgeId>());

    // reserve space for tree and sorted weights
    nodeLayers = std::vector<std::vector<NodeInformation>>(NUMLAYERS);
    edgeLayers = std::vector<std::vector<EdgeInformation>>(NUMLAYERS);
    for (int l = 0; l < NUMLAYERS; l++) {
        ASSERT(parentPointer[l].size() == graphs[l].getNumVertices(), "ParentPointer has size " << parentPointer[l].size() << " but graph has " << graphs[l].getNumVertices() << " vertices");
        ASSERT(edgeParentPointers[l].size() == graphs[l].getNumEdges() * 2, "EdgeParentPointer has size " << edgeParentPointers[l].size() << " but graph has " << graphs[l].getNumEdges() << " edges");

        nodeLayers[l] = std::vector<NodeInformation>(graphs[l].getNumVertices());
        edgeLayers[l] = std::vector<EdgeInformation>(graphs[l].getNumEdges() * 2);
    }

    // assign parent and child nodes
    LOG_INFO("Setting pointers between layers");
    for (int l = 0; l < NUMLAYERS-1; l++)  // only do it for #layer -1
    {
        LOG_INFO("... in layer " << l);
        // assign pointers for nodes
        int nodeLayerSize = parentPointer[l].size();
        for (int i = 0; i < nodeLayerSize; i++) {
            nodeLayers[l][i].parentNode = parentPointer[l][i];             // assign parent
            nodeLayers[l + 1][parentPointer[l][i]].children.push_back(i);  // assign children
        }

        // assign pointers for edges
        int edgeLayerSize = edgeParentPointers[l].size();
        for (int i = 0; i < edgeLayerSize; i++) {
            edgeLayers[l][i].parentEdge = edgeParentPointers[l][i];  // assign parent
            if (edgeParentPointers[l][i] != -1) {
                // some edges don't exists in upper layers. we cant assign children to these
                edgeLayers[l + 1][edgeParentPointers[l][i]].children.push_back(i);  // assign children
            }
        }
    }
    LOG_INFO("Finished building hierarchy");
}

int GraphHierarchy::getNumLayers() const {
    return NUMLAYERS;
}
int GraphHierarchy::getLayerSize(int layer) const {
    return graphs[layer].getNumVertices();
}
