set(target graphLib)

set(HEADERS
        include/graphIO/GraphIO.hpp

        include/graph/GeometricGraphSampler.hpp
        include/graph/Graph.hpp
        include/graph/GraphAlgorithms.hpp
)

set(SOURCES
        src/graphIO/GraphIO.cpp

        src/graph/GeometricGraphSampler.cpp
        src/graph/Graph.cpp
        src/graph/GraphAlgorithms.cpp
)

add_library(${target} STATIC ${HEADERS} ${SOURCES})

target_include_directories(${target} PUBLIC 
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/include/graphIO
        ${CMAKE_CURRENT_LIST_DIR}/include/graph
)

target_link_libraries(${target}
        PUBLIC utilLib wembed_flags)
set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)