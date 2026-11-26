#include "initialEmbedder/UniformEmbedder.hpp"

#include "Toolkit.hpp"

std::vector<std::vector<double> > UniformEmbedder::getInitialEmbedding() {
    if (initEmb.empty()) {
        const size_t vertexCount = graph.getNumVertices();
        const size_t embDim = opts.embeddingDimension;
        const double cubeSide = Toolkit::myPow(static_cast<double>(vertexCount), 1.0 / static_cast<double>(embDim));
        const size_t nodesOnEdge = std::ceil(cubeSide);
        const double edgeInterval = cubeSide / (static_cast<double>(nodesOnEdge) - 1.0);

        initEmb = std::vector(vertexCount, std::vector<double>(embDim));
        for (size_t p = 0; p < vertexCount; p++) {
            //For x dimension it is p % nodesOnEdge * interval, for all others, it is p / (nodesOnEdge * d) * interval
            initEmb[p][0] = static_cast<double>(p % nodesOnEdge) * edgeInterval;
            for (size_t d = 1; d < embDim; d++) {
                initEmb[p][d] = static_cast<double>(p / (nodesOnEdge * d)) * edgeInterval;
            }
        }
    }

    return initEmb;
}
