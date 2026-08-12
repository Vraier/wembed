#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Options.hpp"
#include "wembed.h"

void addOptions(CLI::App& app, Options& opts);

// steps through the embedding manually and writes one csv row per iteration
void calculateEmbeddingWithTrace(wembed::Embedder& embedder, const std::string& tracePath) {
    std::ofstream trace(tracePath);
    if (!trace) {
        std::cerr << "Could not open trace file " << tracePath << std::endl;
        std::exit(1);
    }
    trace << std::setprecision(12);
    trace << "iteration,elapsed_ms,num_vertices,loss_attract,loss_repel,loss_total,learning_rate,rel_displacement,"
             "rel_loss_improvement\n";

    const auto start = std::chrono::steady_clock::now();
    int iteration = 0;
    while (!embedder.isFinished()) {
        embedder.calculateStep();
        iteration++;
        const double elapsedMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const wembed::Loss loss = embedder.getLoss();
        trace << iteration << ',' << elapsedMs << ',' << embedder.getNumVertices() << ',' << loss.attractive << ','
              << loss.repulsive << ',' << loss.total << ',' << embedder.getCurrentLearningRate() << ','
              << embedder.getLastRelDisplacement() << ',' << embedder.getLastRelLossImprovement() << '\n';
    }
}

int main(int argc, char* argv[]) {
    CLI::App app("Embedder CLI");
    Options opts;
    addOptions(app, opts);
    CLI11_PARSE(app, argc, argv);

    if (opts.seed != -1) {
        wembed::setSeed(opts.seed);
    }

    wembed::Graph graph = wembed::graphFromEdgeListFile(opts.graphPath);

    wembed::Embedder embedder = wembed::createEmbedder(graph, opts.embedderOptions);

    if (!opts.inputEmbeddingPath.empty()) {
        auto coords = wembed::readCoordinatesFromFile(
            opts.inputEmbeddingPath, opts.embeddingComment, opts.embeddingDelimiter);
        embedder.setCoordinates(coords);
    }

    if (!opts.tracePath.empty()) {
        calculateEmbeddingWithTrace(embedder, opts.tracePath);
    } else {
        embedder.calculateEmbedding();
    }

    if (opts.showTimings) {
        std::cout << wembed::timingsToString(embedder.getTimings());
    }

    if (!opts.embeddingPath.empty()) {
        embedder.writeCoordinates(opts.embeddingPath);
    }
    return 0;
}

void addOptions(CLI::App& app, Options& opts) {
    wembed::Options& eo = opts.embedderOptions;

    const std::string io = "Input/Output";
    app.add_option("-i,--graph", opts.graphPath, "Path to an edge list")
        ->required()->check(CLI::ExistingFile)->group(io);
    app.add_option("-o,--embedding", opts.embeddingPath, "Path to the output embedding file")
        ->group(io);
    app.add_option("--init-coordinates", opts.inputEmbeddingPath,
                   "Path to a file containing initial coordinates. If empty, coordinates are initialized randomly.")
        ->group(io);
    app.add_flag("--timings", opts.showTimings, "Print timings after embedding")
        ->group(io);
    app.add_option("--trace", opts.tracePath,
                   "Path to a csv file. Writes iteration, wallclock time, graph size, loss and learning rate "
                   "for every step of the embedding.")
        ->group(io);

    const std::string embedding = "Embedding";
    app.add_option("--seed", opts.seed, "Seed used during embedding. '-1' uses time as seed")
        ->capture_default_str()->group(embedding);
    app.add_flag("--layered", eo.layeredEmbedding, "Use layered embedding")
        ->group(embedding);
    app.add_option("--dim", eo.embeddingDimension, "Embedding dimension")
        ->capture_default_str()->group(embedding);
    app.add_option("--dim-hint", eo.dimensionHint, "Dimension hint. Negative values use dim as dimension hint.")
        ->capture_default_str()->group(embedding);
    app.add_flag("--unit-weights", eo.useUnitWeights, "Disable degree-based weights (use unit weights instead)")
        ->group(embedding);
    app.add_option("--index-type", eo.indexType, "Type of spatial index used for the embedding (1=SNN, 2=Sprk)")
        ->capture_default_str()->group(embedding);
    app.add_option("--attraction", eo.attractionScale, "Changes magnitude of attracting forces")
        ->capture_default_str()->group(embedding);
    app.add_option("--repulsion", eo.repulsionScale, "Changes magnitude of repulsing forces")
        ->capture_default_str()->group(embedding);
    app.add_option("--centre,--center", eo.centreScale,
                   "Strength of the centre-pull force. Useful for unconnected graphs (try ~0.01-0.1). "
                   "Default 0 disables it.")
        ->capture_default_str()->group(embedding);
    app.add_option("--expansion", eo.expansionStretch,
                   "Determines how much the embedding is stretched during layer expansion.")
        ->capture_default_str()->group(embedding);
    app.add_option("--iterations", eo.maxIterations, "Maximum number of iterations")
        ->capture_default_str()->group(embedding);
    app.add_option("--optimizer", eo.optimizerType, "Gradient descent optimizer (0=Simple, 1=Adam)")
        ->capture_default_str()->group(embedding);
    app.add_option("--simple-max-displacement", eo.simpleOptMaxDisplacement,
                   "Per-step displacement cap for the Simple optimizer")
        ->capture_default_str()->group(embedding);

    const std::string schedule = "Learning rate schedule";
    app.add_option("--lr-schedule", eo.lrSchedule,
                   "Learning rate schedule (0=ExponentialCooling, 1=LossAdaptive)")
        ->capture_default_str()->group(schedule);
    app.add_option("--speed", eo.learningRate, "Initial learning rate (both schedules)")
        ->capture_default_str()->group(schedule);
    app.add_option("--warmup-steps", eo.warmupSteps,
                   "Linear learning rate ramp-up over the first steps (both schedules)")
        ->capture_default_str()->group(schedule);
    app.add_option("--lr-cooling", eo.lrCoolingFactor,
                   "Per-step multiplicative learning rate decay (schedule 0 only)")
        ->capture_default_str()->group(schedule);
    app.add_option("--lr-decay-factor", eo.lrDecayFactor,
                   "Multiplicative learning rate drop on a decay event (schedule 1 only)")
        ->capture_default_str()->group(schedule);
    app.add_option("--lr-decay-threshold", eo.lrDecayThreshold,
                   "Decay when rate(t) stays below this loss-decrease rate (schedule 1 only)")
        ->capture_default_str()->group(schedule);
    app.add_option("--lr-adapt-patience", eo.lrAdaptPatience,
                   "Consecutive in-zone steps before a decay OR growth event (schedule 1 only)")
        ->capture_default_str()->group(schedule);
    app.add_option("--lr-growth-factor", eo.lrGrowthFactor,
                   "Multiplicative learning rate growth while the loss keeps decreasing fast "
                   "(schedule 1 only; 1.0 disables growth)")
        ->capture_default_str()->group(schedule);
    app.add_option("--lr-growth-threshold", eo.lrGrowthThreshold,
                   "Grow when rate(t) stays above this loss-decrease rate (schedule 1 only)")
        ->capture_default_str()->group(schedule);

    const std::string stopping = "Stopping criterion";
    app.add_option("--stop-criterion", eo.stopCriterion,
                   "Signal that terminates the run (0=Displacement, 1=Loss)")
        ->capture_default_str()->group(stopping);
    app.add_option("--stop-displacement-tol", eo.stopDisplacementTol,
                   "Relative per-step node movement (mean displacement / radius of gyration) below which "
                   "the layout counts as settled (criterion 0)")
        ->capture_default_str()->group(stopping);
    app.add_option("--stop-displacement-patience", eo.stopDisplacementPatience,
                   "Settled steps in a row before stopping (criterion 0)")
        ->capture_default_str()->group(stopping);
    app.add_option("--loss-smoothing", eo.lossSmoothingFactor,
                   "EMA weight of the newest loss sample before the loss-progress monitor sees it "
                   "(1.0 disables smoothing)")
        ->capture_default_str()->group(stopping);
    app.add_option("--loss-rate-window", eo.lossRateWindow,
                   "Steps over which the relative loss-decrease rate rate(t) is measured")
        ->capture_default_str()->group(stopping);
    app.add_option("--stop-loss-tol", eo.stopLossTol,
                   "ftol: converged once rate(t) stays below this relative loss decrease (criterion 1)")
        ->capture_default_str()->group(stopping);
    app.add_option("--stop-loss-patience", eo.stopLossPatience,
                   "Sub-tolerance steps in a row before stopping (criterion 1)")
        ->capture_default_str()->group(stopping);
}
