#include "EdgeSampler.hpp"

#include "Macros.hpp"

bool histComparator(const histEntry& a, const histEntry& b) {
    return a.similarity < b.similarity;
}

histInfo EdgeSampler::sampleHistEntries(const OptionValues& opts, const Graph& graph, Embedding& embedding) {
    const ll N = graph.getNumVertices();
    const ll M = graph.getNumEdges();
    const ll noM = (N * (N - 1) / 2) - M;

    Histogram histogram;
    int numSampledNonEdges = 0;
    int numSampledEdges = 0;

    LOG_INFO( "Sampling a histogram in graph with n:" << N << ", m:" << M << ", no m:" << noM);

    // calculate all edges
    for (int v = 0; v < N; v++) {
        for (int w : graph.getNeighbors(v)) {
            if (w <= v) continue;
            histEntry tmp = getHistEntry(embedding, v, w, true);
            histogram.push_back(tmp);
            numSampledEdges++;
        }
    }

    // sample non edges
    int v = 0;
    int w = 0;
    // scale determine how much more non edges than edges
    double nonEdgeProb = std::min(1.0, opts.edgeSampleScale * M / noM);
    LOG_INFO( "Choosing non-edges with probability " << nonEdgeProb);
    ll totalJumpSize = 0;
    while (true) {
        // calculate how many nodes should be skipped
        ll jumpSize = Rand::geometricVariable(nonEdgeProb) + 1;
        totalJumpSize += jumpSize;

        v = v + ((w + jumpSize) / N);
        w = (w + jumpSize) % N;

        if (v >= N) break;

        // skip  edges
        // only take non-edges where v > w
        std::vector<int> neighbors = graph.getNeighbors(w);
        if (w <= v || std::find(neighbors.begin(), neighbors.end(), v) != neighbors.end()) {
            continue;
        }

        histEntry tmp = getHistEntry(embedding, v, w, false);
        histogram.push_back(tmp);

        numSampledNonEdges++;
    }

    std::sort(histogram.begin(), histogram.end(), histComparator);
    LOG_INFO( "Choose " << numSampledEdges << " edges and " << numSampledNonEdges << " non edges for a total of " << histogram.size() << " node pairs");
    return histInfo{histogram, numSampledEdges, numSampledNonEdges};
}

histEntry EdgeSampler::getHistEntry(Embedding& embedding, NodeId v, NodeId w, bool isEdge) {
    histEntry tmp;
    tmp.v = v;
    tmp.w = w;
    tmp.weightV = -1,
    tmp.weightW = -1,
    tmp.distance = -1,
    tmp.similarity = embedding.getSimilarity(v, w);
    tmp.isEdge = isEdge;
    return tmp;
}

histEntry EdgeSampler::getHistEntry(WeightedGeometric& embedding, NodeId v, NodeId w, bool isEdge) {
    histEntry tmp;
    tmp.v = v;
    tmp.w = w;
    tmp.weightV = embedding.getNodeWeight(v),
    tmp.weightW = embedding.getNodeWeight(w),
    tmp.distance = embedding.getDistance(v, w),
    tmp.similarity = embedding.getSimilarity(v, w);
    tmp.isEdge = isEdge;
    return tmp;
}
