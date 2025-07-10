set(target embeddingLib)

set(HEADERS
        include/dVec/DVec.hpp
        include/dVec/VecList.hpp

        include/embeddingSpace/Embedding.hpp
        include/embeddingSpace/Euclidean.hpp
        include/embeddingSpace/InfNorm.hpp
        include/embeddingSpace/DotProduct.hpp
        include/embeddingSpace/WeightedGeometric.hpp
        include/embeddingSpace/WeightedGeometricInf.hpp
        include/embeddingSpace/WeightedNoDim.hpp
        include/embeddingSpace/Cosine.hpp
        include/embeddingSpace/MercatorEmbedding.hpp

        include/embeddingIO/EmbeddingIO.hpp

        include/partition/LabelPropagation.hpp
        include/partition/GraphHierarchy.hpp

        include/spacialQuery/WeightedIndex.hpp
        include/spacialQuery/SpacialIndex.hpp
        include/spacialQuery/RTree.hpp
        include/spacialQuery/SNNQueries.hpp

        # Embedder
        include/embedder/EmbedderOptions.hpp
        include/embedder/WEmbedEmbedder.hpp
        include/embedder/LayeredEmbedder.hpp

        include/gradientOptimizer/AdamOptimizer.hpp
        include/gradientOptimizer/SimpleOptimizer.hpp
)

set(SOURCES
        src/embeddingSpace/Euclidean.cpp
        src/embeddingSpace/InfNorm.cpp
        src/embeddingSpace/DotProduct.cpp
        src/embeddingSpace/WeightedGeometric.cpp
        src/embeddingSpace/WeightedGeometricInf.cpp
        src/embeddingSpace/WeightedNoDim.cpp
        src/embeddingSpace/Cosine.cpp
        src/embeddingSpace/Poincare.cpp
        src/embeddingSpace/MercatorEmbedding.cpp
        
        src/embeddingIO/EmbeddingIO.cpp

        src/partition/LabelPropagation.cpp
        src/partition/GraphHierarchy.cpp

        src/spacialQuery/WeightedIndex.cpp
        src/spacialQuery/SNNQueries.cpp

        # Embedder
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

target_link_libraries(${target} PUBLIC wembed_flags graphLib OpenMP::OpenMP_CXX snn)
set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
