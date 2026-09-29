* Move internals (Graph, EmbedderInterface, ...) into a `wembed::detail` namespace
* Fix embedder using non const graph reference so `wembed.cpp` doesn't need const_cast
* Remove dead EmbedderOptions fields (optimizerType, weightPenalty, lpNorm, weightLearningRate, dumpWeights, WeightType::Original)
* remove code duplication of options, timingResults, SpatialIndex in cpp library interface
    * replace inline map with constexpr 
* LossAdaptive: guard the growth counter during the rate warmup (rate = +inf); decide whether a negative rate (loss bump after expansion) should hold instead of decay
