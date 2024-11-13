#pragma once

#include <memory>

// #include "AbstractLayerEmbedder.hpp"
#include "EmbedderInterface.hpp"
#include "EmbedderOptions.hpp"
#include "Graph.hpp"

/**
 * Factory class to create Embedder objects
 *
 * TODO: implement layerEmbedders
 * TODO: use namespace
 */
class EmbedderFactory {
   public:
    static std::unique_ptr<EmbedderInterface> constructSimpleEmbedder(EmbedderOptions options, Graph g);
};

// static AbstractLayerEmbedder* constructLayerEmbedder(EmbedderOptions options, GraphHierarchy* hierarchy,
//                                                      int currLayer);
