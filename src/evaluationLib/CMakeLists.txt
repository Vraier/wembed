add_library(evaluationLib STATIC
    src/metrics/ConfigParser.cpp
    src/metrics/EdgeDetection.cpp
    src/metrics/EdgeSampler.cpp
    src/metrics/GeneralGraphInfo.cpp
    src/metrics/Metric.cpp
    src/metrics/NodeSampler.cpp
    src/metrics/Reconstruction.cpp
    src/metrics/TimeParser.cpp
)
target_include_directories(evaluationLib PUBLIC include/metrics)
target_link_libraries(evaluationLib PUBLIC wembed wembed_internal)
