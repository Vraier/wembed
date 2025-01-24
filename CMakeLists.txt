cmake_minimum_required(VERSION 3.15...3.27)

project(WEmbed VERSION 0.0.1 LANGUAGES CXX)

# define compiler flags for project
# use interface library to not force the flags on the user
add_library(wembed_flags INTERFACE)
target_compile_features(wembed_flags INTERFACE cxx_std_20)
target_compile_options(wembed_flags INTERFACE -Wall -Wextra -Wno-sign-compare -g -DBOOST_ALLOW_DEPRECATED_HEADERS -fno-omit-frame-pointer)

#target_compile_options(wembed_flags INTERFACE "$<BUILD_INTERFACE:$<$<CONFIG:RELEASE>:-mtune=native -march=native>>") # for some reason compiling with pip does not work with this
target_compile_options(wembed_flags INTERFACE "$<BUILD_INTERFACE:$<$<CONFIG:DEBUG>:-fsanitize=address,undefined>>") 
target_link_options(wembed_flags INTERFACE "$<BUILD_INTERFACE:$<$<CONFIG:DEBUG>:-fsanitize=address,undefined>>")

# Dependencies
# openmp 
find_package(OpenMP REQUIRED)
# Eigen
find_package(Eigen3 REQUIRED)
# We only need the header libraries of boost but cmake can't require only the header libraries (?)
#find_package(Boost 1.74 REQUIRED)

include(FetchContent)
# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
)
# CLI11
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.3.2
)
# Girg generator
FetchContent_Declare(
    girgs 
    GIT_REPOSITORY https://github.com/chistopher/girgs.git
)

# Fetch pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(googletest cli11 girgs pybind11)

# try to find sfml, if it is not avaliable we will not build the drawLib
find_package(SFML 2.5 COMPONENTS window graphics system audio QUIET)
if(SFML_FOUND)
        message(STATUS "SFML found, embedding animation will be available")
else()
        message(STATUS "SFML not found, no embedding animation will be available")
endif()

# determine the directories for the binary files
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# additional compile options
option(EMBEDDING_USE_ASSERTIONS "Explicitely activates assertions (even in release mode)." OFF)
if(EMBEDDING_USE_ASSERTIONS)
        add_compile_definitions(EMBEDDING_USE_ASSERTIONS)
endif(EMBEDDING_USE_ASSERTIONS)

enable_testing()

# export compile_commands.json, which can be used by editors for auto
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)
add_subdirectory(tests)
