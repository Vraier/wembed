FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.17.0
    GIT_SHALLOW    ON
    SYSTEM EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(googletest)

add_executable(UnitTests
    TestDVec.cpp
    TestDeterminism.cpp
    TestGraph.cpp
    TestGraphAlgorithm.cpp
    TestGeometricGraphSampler.cpp
    TestHierarchy.cpp
    TestLossFunction.cpp
    TestMetrics.cpp
    TestSpatialIndex.cpp
    TestTimings.cpp
    TestWeightedIndex.cpp
)
target_link_libraries(UnitTests PRIVATE GTest::gtest_main evaluationLib)

include(GoogleTest)
gtest_discover_tests(UnitTests)
