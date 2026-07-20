# Libraries
add_subdirectory(graphLib)
add_subdirectory(utilLib)
add_subdirectory(embeddingLib)
add_subdirectory(SNN)
add_subdirectory(sprk)

if(NOT WEMBED_AS_LIBRARY)
    add_subdirectory(evaluationLib)
    add_subdirectory(drawLib)
endif()

# Public C++ library
add_library(wembedLib STATIC "wembed.cpp")
target_include_directories(wembedLib PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(wembedLib PRIVATE wembed_flags graphLib embeddingLib)
set_target_properties(wembedLib PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# Executables
if(NOT WEMBED_AS_LIBRARY)
    add_subdirectory(cli_wembed)
    add_subdirectory(cli_evaluator)
    add_subdirectory(cli_generator)
endif()
