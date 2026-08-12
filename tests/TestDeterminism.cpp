#include <gtest/gtest.h>

#include <vector>

#include "Rand.hpp"
#include "WembedEmbedder.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// A small connected graph with enough vertices to spread across several threads.
Graph makeRingGraph(int n) {
    std::vector<std::pair<int, int>> edges;
    for (int v = 0; v < n; v++) {
        edges.emplace_back(v, (v + 1) % n);
        edges.emplace_back(v, (v + 7) % n);  // chords, keeps degrees > 1
    }
    return Graph(edges);
}

struct RunResult {
    std::vector<std::vector<double>> coordinates;
    double loss;
    int iterations;
};

// Runs the embedder and returns the resulting coordinates, loss and iteration
// count. If steps >= 0 it runs exactly that many steps; if steps < 0 it runs to
// convergence (until isFinished()), which exercises the stopping criterion.
// coincidentStart places all nodes at the same position.
// otherwise the constructor's random initialisation is used. for the stopping criterion.
RunResult runEmbedding(const EmbedderOptions& opts, int numThreads, int steps, bool coincidentStart) {
#ifdef _OPENMP
    omp_set_num_threads(numThreads);
#else
    (void)numThreads;
#endif
    Rand::setSeed(1234);

    Graph g = makeRingGraph(64);
    WembedEmbedder embedder(g, opts);

    if (coincidentStart) {
        std::vector<std::vector<double>> start(g.getNumVertices(),
                                               std::vector<double>(opts.embeddingDimension, 0.0));
        embedder.setCoordinates(start);
    }

    int iterations = 0;
    if (steps < 0) {
        while (!embedder.isFinished()) {
            embedder.calculateStep();
            iterations++;
        }
    } else {
        for (int i = 0; i < steps; i++) {
            embedder.calculateStep();
            iterations++;
        }
    }
    return {embedder.getCoordinates(), embedder.getLoss().total, iterations};
}

void expectExactlyEqual(const std::vector<std::vector<double>>& a,
                        const std::vector<std::vector<double>>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t v = 0; v < a.size(); v++) {
        ASSERT_EQ(a[v].size(), b[v].size());
        for (size_t d = 0; d < a[v].size(); d++) {
            // determinism means bit-for-bit identical, not merely close
            EXPECT_EQ(a[v][d], b[v][d]) << "mismatch at node " << v << " dim " << d;
        }
    }
}

EmbedderOptions baseOptions() {
    EmbedderOptions opts;
    opts.embeddingDimension = 2;
    opts.maxIterations = 1000;  
    return opts;
}

}  // namespace

// Two runs with the same seed and the same (multi-)thread count must agree.
TEST(Determinism, RepeatedMultiThreadedRunsMatch) {
    EmbedderOptions opts = baseOptions();
    auto first = runEmbedding(opts, 4, 25, true);
    auto second = runEmbedding(opts, 4, 25, true);
    expectExactlyEqual(first.coordinates, second.coordinates);
}

// The embedding must not depend on how many threads OpenMP uses.
TEST(Determinism, ResultIndependentOfThreadCount) {
    EmbedderOptions opts = baseOptions();
    auto single = runEmbedding(opts, 1, 25,true);
    auto multi = runEmbedding(opts, 4, 25,true);
    expectExactlyEqual(single.coordinates, multi.coordinates);
}

// The loss that drives the stopping criterion must also be thread-count
// independent, otherwise a run could stop at a different iteration on a different
// machine. This exercises the deterministic loss reduction.
TEST(Determinism, LossIndependentOfThreadCount) {
    EmbedderOptions opts = baseOptions();
    auto single = runEmbedding(opts, 1, 25, true);
    auto multi = runEmbedding(opts, 4, 25, true);
    EXPECT_EQ(single.loss, multi.loss);
}

// running to convergence from a realistic (random) start must stop at
// the same iteration (and produce the same embedding) regardless of thread count.
// This exercises the loss-based stopping criterion under the ftol semantics: the
// windowed relative loss-decrease rate rate(t) is a deterministic function of the
// deterministically-reduced loss stream plus a fixed-size buffer, so the stop
// iteration is thread-count independent.
TEST(Determinism, LossStoppingIterationIndependentOfThreadCount) {
    EmbedderOptions opts = baseOptions();
    opts.stopCriterion = StopCriterionType::Loss;
    opts.lossRateWindow = 10;    // rate measured over 10 steps
    opts.stopLossTol = 1e-2;     // ftol: stop when the window relative decrease stays below 1%
    opts.stopLossPatience = 10;  // sub-tolerance steps in a row before stopping
    opts.maxIterations = 2000;

    auto single = runEmbedding(opts, 1, -1, false);
    auto multi = runEmbedding(opts, 4, -1, false);

    EXPECT_GT(single.iterations, opts.stopLossPatience);
    EXPECT_LT(single.iterations, opts.maxIterations);

    EXPECT_EQ(single.iterations, multi.iterations);
    expectExactlyEqual(single.coordinates, multi.coordinates);
}

// The LossAdaptive schedule reads the same windowed loss-decrease rate rate(t)
// and adjusts the learning rate from it. Those adjustments are deterministic, so
// a fixed-length run must be bit-for-bit identical across thread counts.
TEST(Determinism, LossAdaptiveScheduleIndependentOfThreadCount) {
    EmbedderOptions opts = baseOptions();
    opts.lrScheduleType = LRScheduleType::LossAdaptive;
    opts.lossRateWindow = 10;
    opts.lrDecayThreshold = 1e-2;
    opts.lrDecayFactor = 0.5;
    opts.lrGrowthThreshold = 1e-1;
    opts.lrGrowthFactor = 1.0;  // growth off: pure plateau decay
    opts.lrAdaptPatience = 5;   // a few decay events within the run

    // 60 steps > lossRateWindow + a few adapt cycles, so at least one decay fires
    auto single = runEmbedding(opts, 1, 60, false);
    auto multi = runEmbedding(opts, 4, 60, false);

    expectExactlyEqual(single.coordinates, multi.coordinates);
    EXPECT_EQ(single.loss, multi.loss);
}

// Same guarantee for the default displacement-based stopping criterion: the
// relative-displacement signal is reduced deterministically, so the run must
// stop at the same iteration on any thread count.
TEST(Determinism, DisplacementStoppingIterationIndependentOfThreadCount) {
    EmbedderOptions opts = baseOptions();
    opts.stopCriterion = StopCriterionType::Displacement;
    opts.stopDisplacementTol = 1e-3;
    opts.stopDisplacementPatience = 5;
    opts.maxIterations = 5000;

    auto single = runEmbedding(opts, 1, -1, false);
    auto multi = runEmbedding(opts, 4, -1, false);

    EXPECT_GT(single.iterations, opts.stopDisplacementPatience);
    EXPECT_LT(single.iterations, opts.maxIterations);

    EXPECT_EQ(single.iterations, multi.iterations);
    expectExactlyEqual(single.coordinates, multi.coordinates);
}
