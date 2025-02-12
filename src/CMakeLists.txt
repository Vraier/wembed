# Libraries
add_subdirectory(graphLib)
add_subdirectory(utilLib)
add_subdirectory(embeddingLib)
add_subdirectory(evaluationLib)
add_subdirectory(drawLib)

# Executables
add_subdirectory(cli_wembed)
add_subdirectory(cli_evaluator)
add_subdirectory(cli_generator)

# Python Bindings
add_subdirectory(bindings)