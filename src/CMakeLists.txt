# Libraries
add_subdirectory(graphLib)
add_subdirectory(utilLib)
add_subdirectory(embeddingLib)
add_subdirectory(SNN)

if(NOT WEMBED_AS_LIBRARY)
    add_subdirectory(evaluationLib)
    add_subdirectory(drawLib)
endif()

# Executables
if(NOT WEMBED_AS_LIBRARY)
    add_subdirectory(cli_wembed)
    add_subdirectory(cli_evaluator)
    add_subdirectory(cli_generator)
endif()
