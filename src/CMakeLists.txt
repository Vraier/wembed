if(WEMBED_USE_SPRK)
    add_subdirectory(sprk)
endif()

# The wembed library. Its public interface is include/wembed.h,
# utilLib, graphLib and embeddingLib are source folders of this one target.
add_library(wembed STATIC
    wembed.cpp

    utilLib/src/FileOperations.cpp
    utilLib/src/Rand.cpp
    utilLib/src/StringManipulation.cpp
    utilLib/src/Timings.cpp
    utilLib/src/Toolkit.cpp

    graphLib/src/graph/GeometricGraphSampler.cpp
    graphLib/src/graph/Graph.cpp
    graphLib/src/graph/GraphAlgorithms.cpp
    graphLib/src/graphIO/GraphIO.cpp

    embeddingLib/src/embedder/ConvergenceMonitor.cpp
    embeddingLib/src/embedder/DisplacementMonitor.cpp
    embeddingLib/src/embedder/LayeredEmbedder.cpp
    embeddingLib/src/embedder/WembedEmbedder.cpp
    embeddingLib/src/embeddingIO/EmbeddingIO.cpp
    embeddingLib/src/embeddingSpace/Additive.cpp
    embeddingLib/src/embeddingSpace/Cosine.cpp
    embeddingLib/src/embeddingSpace/DotProduct.cpp
    embeddingLib/src/embeddingSpace/Euclidean.cpp
    embeddingLib/src/embeddingSpace/InfNorm.cpp
    embeddingLib/src/embeddingSpace/MercatorEmbedding.cpp
    embeddingLib/src/embeddingSpace/Poincare.cpp
    embeddingLib/src/embeddingSpace/WeightedGeometric.cpp
    embeddingLib/src/embeddingSpace/WeightedGeometricInf.cpp
    embeddingLib/src/embeddingSpace/WeightedNoDim.cpp
    embeddingLib/src/gradientOptimizer/AdamOptimizer.cpp
    embeddingLib/src/gradientOptimizer/LRScheduler.cpp
    embeddingLib/src/gradientOptimizer/SimpleOptimizer.cpp
    embeddingLib/src/partition/GraphHierarchy.cpp
    embeddingLib/src/partition/LabelPropagation.cpp
    embeddingLib/src/spacialQuery/KdTreeQueries.cpp
    embeddingLib/src/spacialQuery/WeightedIndex.cpp
)
add_library(wembed::wembed ALIAS wembed)

# Everything behind wembed.h. Tests and the research CLIs use these headers directly, users of the library do not.
add_library(wembed_internal INTERFACE)
target_include_directories(wembed_internal INTERFACE
    utilLib/include
    graphLib/include/graph
    graphLib/include/graphIO
    embeddingLib/include/dVec
    embeddingLib/include/embedder
    embeddingLib/include/embeddingIO
    embeddingLib/include/embeddingSpace
    embeddingLib/include/gradientOptimizer
    embeddingLib/include/partition
    embeddingLib/include/spacialQuery
)
target_link_libraries(wembed_internal INTERFACE wembed_flags OpenMP::OpenMP_CXX)
if(WEMBED_USE_SPRK)
    target_sources(wembed PRIVATE embeddingLib/src/spacialQuery/SprkQueries.cpp)
    target_compile_definitions(wembed_internal INTERFACE WEMBED_HAS_SPRK)
    target_link_libraries(wembed_internal INTERFACE sprk_ffi-static)
endif()

target_include_directories(wembed PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(wembed PRIVATE wembed_internal)
# wembed ends up inside the shared python module
set_target_properties(wembed PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(WEMBED_BUILD_TOOLS)
    FetchContent_Declare(
        cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG        v2.5.0
        GIT_SHALLOW    ON
        SYSTEM EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(cli11)

    add_subdirectory(evaluationLib)
    add_subdirectory(cli_wembed)
    add_subdirectory(cli_evaluator)
    add_subdirectory(cli_generator)
endif()
