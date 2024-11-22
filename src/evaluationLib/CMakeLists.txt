set(target evaluationLib)

set(HEADERS
        include/metrics/ConfigParser.hpp
        include/metrics/EdgeDetection.hpp
        include/metrics/EdgeSampler.hpp
        include/metrics/GeneralGraphInfo.hpp
        include/metrics/Metric.hpp
        include/metrics/NodeSampler.hpp
        include/metrics/Reconstruction.hpp
        include/metrics/TimeParser.hpp
)

set(SOURCES
        src/metrics/ConfigParser.cpp
        src/metrics/EdgeDetection.cpp
        src/metrics/EdgeSampler.cpp
        src/metrics/GeneralGraphInfo.cpp
        src/metrics/Metric.cpp
        src/metrics/NodeSampler.cpp
        src/metrics/Reconstruction.cpp
        src/metrics/TimeParser.cpp
)

add_library(${target} STATIC ${HEADERS} ${SOURCES})

target_include_directories(${target} PUBLIC 
        ${CMAKE_CURRENT_LIST_DIR}/include/metrics
)

target_link_libraries(${target} PUBLIC wembed_flags utilLib graphLib embeddingLib)
set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)