set(TEST_FILES
    TestDVec.cpp
    TestDeterminism.cpp
    TestGraph.cpp
    TestGraphAlgorithm.cpp
    TestGeometricGraphSampler.cpp
    #TestHierarchy.cpp
    TestMetrics.cpp
    TestTimings.cpp
)

add_executable(UnitTests ${TEST_FILES})
target_link_libraries(UnitTests GTest::gtest_main embeddingLib graphLib evaluationLib)

include(GoogleTest)
gtest_add_tests(TARGET UnitTests)
