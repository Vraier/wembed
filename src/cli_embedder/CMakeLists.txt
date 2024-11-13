set(target cli_embedder)

add_executable(${target})

target_include_directories(${target} PRIVATE 
        "include/util")

target_sources(${target} PRIVATE
        "src/main.cpp"

        "src/util/EmbOptions.cpp"
)

target_link_libraries(${target} PRIVATE embeddingLib drawLib girgs::girgs)