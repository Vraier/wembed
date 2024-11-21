set(target evaluator)

add_executable(${target})

target_include_directories(${target} PRIVATE 
        "include"
        "include/metrics")

target_sources(${target} PRIVATE
        "src/main.cpp"
                
        "src/metrics/ConfigParser.cpp"
        "src/metrics/TimeParser.cpp"
        "src/metrics/GeneralGraphInfo.cpp"
        "src/metrics/EdgeDetection.cpp"
        "src/metrics/EdgeSampler.cpp"
        "src/metrics/Metric.cpp"
        "src/metrics/NodeSampler.cpp"
        "src/metrics/Reconstruction.cpp")

target_link_libraries(${target} 
        PRIVATE graphLib embeddingLib CLI11::CLI11)