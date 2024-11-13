set(TEST_FILES
    TestDVec.cpp
    TestGraph.cpp
    TestGraphAlgorithm.cpp
    TestGeometricGraphSampler.cpp
    #TestHierarchy.cpp
    TestWeightedRTree.cpp
    TestTimings.cpp
)

add_executable(UnitTests ${TEST_FILES})
target_link_libraries(UnitTests GTest::gtest_main embeddingLib graphLib)

include(GoogleTest)
gtest_add_tests(TARGET UnitTests)
