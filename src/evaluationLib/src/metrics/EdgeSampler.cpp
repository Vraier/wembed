#include "EdgeSampler.hpp"

#include "Macros.hpp"

bool histComparator(const histEntry& a, const histEntry& b) { return a.similarity < b.similarity; }

histInfo EdgeSampler::sampleHistEntries(const Graph& graph, std::shared_ptr<Embedding> embedding,
                                        flt_t sampleingScale) {
    const ll N = graph.getNumVertices();
    const ll M = graph.getNumEdges();
    const ll maxM = (N * (N - 1) / 2);
    const ll noM = maxM - M;

    std::vector<histEntry> histogram;
    int numSampledNonEdges = 0;
    int numSampledEdges = 0;

    // calculate all edges
    for (int v = 0; v < N; v++) {
        for (int w : graph.getNeighbors(v)) {
            if (w <= v) continue;
            histEntry tmp{embedding->getSimilarity(v, w), v, w, true};
            histogram.push_back(tmp);
            numSampledEdges++;
        }
    }

    // sample non edges
    int v = 0;
    int w = 0;
    // scale determines how much more non edges than edges
    flt_t nonEdgeProb = std::min(flt_t{1.0}, sampleingScale * static_cast<flt_t>(M) / static_cast<flt_t>(noM));
    ll totalJumpSize = 0;
    while (true) {
        // calculate how many nodes should be skipped
        ll jumpSize = Rand::geometricVariable(nonEdgeProb) + 1;
        totalJumpSize += jumpSize;

        v = v + ((w + jumpSize) / N);
        w = (w + jumpSize) % N;

        if (v >= N) break;

        // skip edges
        // only take non-edges where v > w
        std::vector<int> neighbors = graph.getNeighbors(w);
        if (w <= v || graph.areNeighbors(v, w)) {
            continue;
        }

        histEntry tmp{embedding->getSimilarity(v, w), v, w, false};
        histogram.push_back(tmp);
        numSampledNonEdges++;
    }

    std::sort(histogram.begin(), histogram.end(), histComparator);
    return histInfo{histogram, numSampledEdges, numSampledNonEdges};
}
