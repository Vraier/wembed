cmake_minimum_required(VERSION 3.15...3.27)

project(WEmbed VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# export compile_commands.json, which can be used by editors for auto
# completion (e.g., using lsp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-compare -g -DBOOST_ALLOW_DEPRECATED_HEADERS -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=native -march=native") #-U_FORTIFY_SOURCE
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}") # -fsanitize=address,undefined")

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
enable_testing()

# additional compile options
option(EMBEDDING_USE_ASSERTIONS "Explicitely activates assertions (even in release mode)." OFF)
if(EMBEDDING_USE_ASSERTIONS)
        add_compile_definitions(EMBEDDING_USE_ASSERTIONS)
endif(EMBEDDING_USE_ASSERTIONS)

# determine the directories for the binary files
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

add_subdirectory(src)
add_subdirectory(tests)
