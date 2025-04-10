cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project("SNN"
    DESCRIPTION "snn: A lightweight fast exact radius query algorithm"
    VERSION 1.0)

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

file(GLOB_RECURSE SRC_FILES ${SOURCE_DIR}/*.cpp)

add_library(snn STATIC ${SRC_FILES})

target_include_directories(snn PUBLIC ${INCLUDE_DIR})
set_target_properties(snn PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

