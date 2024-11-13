#pragma once

#include "Graph.hpp"
#include "EmbOptions.hpp"
#include "Partitioner.hpp"

// gets a graph, writes it to edge list, calls python script and converts to output to a partition tree
class LouvainNxPartitioner : public Partitioner {
   public:
    virtual NodeParentPointer coarsenAllLayers(OptionValues options, const EmbeddedGraph& g);

   private:
    NodeParentPointer readInFileData(std::string filePath);
};
