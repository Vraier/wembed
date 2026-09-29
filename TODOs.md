* Rename NewWembedEmbedder to WembedEmbedder
* remove bipartite support
* Move internals (Graph, EmbedderInterface, ...) into a `wembed::detail` namespace
* Fix embedder using non const graph reference so `wembed.cpp` doesn't need const_cast
* Remove dead EmbedderOptions fields (optimizerType, weightPenalty, lpNorm, weightLearningRate, dumpWeights, WeightType::Original)
* remove code duplication of options, timingResults, SpatialIndex in cpp library interface
    * replace inline map with constexpr 
* get cmake clean up (maybe take a refcmake project like chris)
* check that embedder is deterministic and has no race conditions (like in the rng generator)
* add a simple KD-Tree datastructure to make build process easier* remove StopDisplacement (+ stopDisplacementTol/Patience, DisplacementMonitor): the loss stop beat it at every cost level on 42 graphs (wembed_experiments tune_stop_val)
* LossAdaptive: guard the growth counter during the rate warmup (rate = +inf); decide whether a negative rate (loss bump after expansion) should hold instead of decay
