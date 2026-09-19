cmake_minimum_required(VERSION 3.28...4.1)

project(WEmbed VERSION 0.2.0 LANGUAGES CXX)

# The wembed library is always built. The tools default to OFF when wembed is part of another project.
option(WEMBED_BUILD_TOOLS "Build the command line tools and unit tests." ${PROJECT_IS_TOP_LEVEL})
option(WEMBED_BUILD_PYTHON "Build the python module." OFF)
option(WEMBED_USE_SPRK "Use the sprk tree as spatial index (needs Rust). If OFF, a bundled KD-tree is used instead." ON)
option(EMBEDDING_USE_ASSERTIONS "Explicitely activates assertions (even in release mode)." OFF)

# Compiler flags for our own targets.
# The wembed library links this PRIVATE, so the flags are not forced on its users.
# Sanitizers and profiling flags are not set here, see CMakePresets.json.
add_library(wembed_flags INTERFACE)
target_compile_features(wembed_flags INTERFACE cxx_std_20)
target_compile_options(wembed_flags INTERFACE -Wall -Wextra -Wno-sign-compare)
if(EMBEDDING_USE_ASSERTIONS)
    target_compile_definitions(wembed_flags INTERFACE EMBEDDING_USE_ASSERTIONS)
endif()

find_package(OpenMP REQUIRED)
include(FetchContent)

# Only touch global settings when wembed is the project being built, not when it is a subproject.
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

add_subdirectory(src)
if(WEMBED_BUILD_TOOLS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(WEMBED_BUILD_PYTHON)
    add_subdirectory(python)
endif()
