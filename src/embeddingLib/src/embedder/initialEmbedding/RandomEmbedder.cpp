#include "initialEmbedder/RandomEmbedder.hpp"

#include "Rand.hpp"
#include "Toolkit.hpp"

std::vector<std::vector<double>> RandomEmbedder::getInitialEmbedding() {
    if (initEmb.empty()) {
        const size_t vertexCount = graph.getNumVertices();
        const size_t embDim = opts.embeddingDimension;
        const double cubeSide = Toolkit::myPow(static_cast<double>(vertexCount), 1.0 / static_cast<double>(embDim));

        initEmb = std::vector(vertexCount, std::vector<double>(embDim));
        for (auto& pos : initEmb) {
            for (auto& c : pos) {
                c = Rand::randomDouble(0, cubeSide);
            }
        }
    }

    return initEmb;
}
