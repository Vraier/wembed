set(target embeddingLib)

set(HEADERS
        include/dVec/DVec.hpp
        include/dVec/VecList.hpp

        include/embeddingSpace/Embedding.hpp
        include/embeddingSpace/Euclidean.hpp
        include/embeddingSpace/DotProduct.hpp
        include/embeddingSpace/WeightedGeometric.hpp
        include/embeddingSpace/WeightedGeometricInf.hpp
        include/embeddingSpace/WeightedNoDim.hpp
        include/embeddingSpace/Cosine.hpp
        include/embeddingSpace/MercatorEmbedding.hpp

        include/embeddedGraph/EmbeddedGraph.hpp

        include/embeddingIO/EmbeddingIO.hpp

        include/partition/LabelPropagation.hpp
        include/partition/GraphHierarchy.hpp

        include/spacialQuery/SpacialIndex.hpp
        include/spacialQuery/RTree.hpp
        include/spacialQuery/WeightedRTree.hpp

        # Embedder
        #include/embedder/EmbedderFactory.hpp
        include/embedder/EmbedderOptions.hpp

        include/embedder/AbstractSimpleEmbedder.hpp
        #include/embedder/AbstractLayerEmbedder.hpp
        #include/embedder/HierarchyEmbedder.hpp
        #include/embedder/KHierarchyEmbedder.hpp
        #include/embedder/WeightedSpringEmbedder.hpp
        #include/embedder/WeightedStressEmbedder.hpp
        #include/embedder/FruchtermannLayerEmbedder.hpp
        #include/embedder/LinearLayerEmbedder.hpp
        #include/embedder/SigmoidLayerEmbedder.hpp
        #include/embedder/LocalEmbedder.hpp
        #include/embedder/SigmoidEmbedder.hpp
        #include/embedder/FruchtermannWeightEmbedder.hpp
        include/embedder/SimpleEmbedders.hpp
        include/embedder/SimpleSamplingEmbedder.hpp
        include/embedder/WEmbedEmbedder.hpp
        include/embedder/LayeredEmbedder.hpp

        include/gradientOptimizer/AdamOptimizer.hpp
        include/gradientOptimizer/SimpleOptimizer.hpp
)

set(SOURCES
        src/embeddingSpace/Euclidean.cpp
        src/embeddingSpace/DotProduct.cpp
        src/embeddingSpace/WeightedGeometric.cpp
        src/embeddingSpace/WeightedGeometricInf.cpp
        src/embeddingSpace/WeightedNoDim.cpp
        src/embeddingSpace/Cosine.cpp
        src/embeddingSpace/Poincare.cpp
        src/embeddingSpace/MercatorEmbedding.cpp
        
        src/embeddedGraph/EmbeddedGraph.cpp

        src/embeddingIO/EmbeddingIO.cpp

        src/partition/LabelPropagation.cpp
        src/partition/GraphHierarchy.cpp

        src/spacialQuery/WeightedRTree.cpp

        # Embedder
        #src/embedder/EmbedderFactory.cpp
        src/embedder/EmbedderOptions.cpp

        src/embedder/AbstractSimpleEmbedder.cpp
        #src/embedder/AbstractLayerEmbedder.cpp
        #src/embedder/HierarchyEmbedder.cpp
        #src/embedder/KHierarchyEmbedder.cpp

        #src/embedder/WeightedSpringEmbedder.cpp
        #src/embedder/WeightedStressEmbedder.cpp

        #src/embedder/FruchtermannLayerEmbedder.cpp
        #src/embedder/LinearLayerEmbedder.cpp
        #src/embedder/SigmoidLayerEmbedder.cpp

        #src/embedder/LocalEmbedder.cpp
        #src/embedder/SigmoidEmbedder.cpp
        #src/embedder/FruchtermannWeightEmbedder.cpp

        src/embedder/SimpleEmbedders.cpp
        src/embedder/SimpleSamplingEmbedder.cpp
        src/embedder/WEmbedEmbedder.cpp
        src/embedder/LayeredEmbedder.cpp

        src/gradientOptimizer/AdamOptimizer.cpp
        src/gradientOptimizer/SimpleOptimizer.cpp
)

add_library(${target} STATIC ${HEADERS} ${SOURCES})

target_include_directories(${target} PUBLIC 
        ${CMAKE_CURRENT_LIST_DIR}/include/dVec
        ${CMAKE_CURRENT_LIST_DIR}/include/embeddingSpace
        ${CMAKE_CURRENT_LIST_DIR}/include/embedder
        ${CMAKE_CURRENT_LIST_DIR}/include/embeddedGraph
        ${CMAKE_CURRENT_LIST_DIR}/include/gradientOptimizer
        ${CMAKE_CURRENT_LIST_DIR}/include/partition
        ${CMAKE_CURRENT_LIST_DIR}/include/embeddingIO
        ${CMAKE_CURRENT_LIST_DIR}/include/spacialQuery)

target_link_libraries(${target} PUBLIC wembed_flags graphLib OpenMP::OpenMP_CXX)
set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
