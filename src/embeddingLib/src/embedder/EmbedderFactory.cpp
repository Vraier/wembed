#include "EmbedderFactory.hpp"

#include "SimpleEmbedders.hpp"
#include "SimpleSamplingEmbedder.hpp"

std::unique_ptr<EmbedderInterface> EmbedderFactory::constructSimpleEmbedder(EmbedderOptions options, Graph g) {
    EmbedderType type = options.embedderType;
    ASSERT(type >= 7 && type <= 18);
    switch (type) {
        //case sFruchtermann:
        //    return new FruchtermannWeightEmbedder(g, options);
        //case sLocal:
        //    return new LocalEmbedder(g, options);
        //case sSigmoid:
        //    return new SigmoidEmbedder(g, options);
        case slowF2V:
            return std::make_unique<SimpleF2VEmbedder>(g, options);
        case slowFruchtermann:
            return std::make_unique<SimpleFruchtermannEmbedder>(g, options);
        case slowSigmoidEuclidean:
            return std::make_unique<SimpleSigmoidEuclideanEmbedder>(g, options);
        case slowMaxent:
            return std::make_unique<SimpleMaxentEmbedder>(g, options);
        case slowF2VNormed:
            return std::make_unique<SimpleF2VNormedEmbedder>(g, options);
        case slowSigmoidWeighted:
            return std::make_unique<SimpleSigmoidWeightedEmbedder>(g, options);
        case slowSigmoidNoDim:
            return std::make_unique<SimpleSigmoidNoDimEmbedder>(g, options);
        case slowLinear:
            return std::make_unique<SimpleLinearEmbedder>(g, options);
        case slowSampling:
            return std::make_unique<SimpleSamplingEmbedder>(g, options);
        default:
            LOG_ERROR( "Unknown embedder type " << type);
            return nullptr;
    }
}

//AbstractLayerEmbedder* EmbedderFactory::constructLayerEmbedder(OptionValues options, GraphHierarchy* hierarchy,
//                                                               int currLayer) {
//    EmbedderType type = EmbedderType(options.embedderOptions.embedderType);
//    ASSERT(type >= 0 && type <= 4);
//    switch (type) {
//        case hFruchtermann:
//            return new FruchtermannLayerEmbedder(options, hierarchy, currLayer);
//        case hSigmoid:
//            return new SigmoidLayerEmbedder(options, hierarchy, currLayer);
//        case hFruchtermannSigmoid:
//            if (currLayer > 0) {
//                return new FruchtermannLayerEmbedder(options, hierarchy, currLayer);
//            } else {
//                return new SigmoidLayerEmbedder(options, hierarchy, currLayer);
//            }
//        case hLinear:
//            return new LinearLayerEmbedder(options, hierarchy, currLayer);
//        case hFruchtermannLinear:
//            if (currLayer > 0) {
//                return new FruchtermannLayerEmbedder(options, hierarchy, currLayer);
//            } else {
//                return new LinearLayerEmbedder(options, hierarchy, currLayer);
//            }
//        default:
//            LOG_ERROR( "Unknown embedder type " << type);
//            return nullptr;
//    }
//}