set(HEADERS
    include/NNDescent.hpp
)

add_library(knn INTERFACE)

target_include_directories(knn INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)