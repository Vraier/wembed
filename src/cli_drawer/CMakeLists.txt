set(target cli_drawer)


add_executable(${target})

target_include_directories(${target} PRIVATE 
        ".")

target_sources(${target} PRIVATE
        "main.cpp"
        "DrawOptions.cpp")

target_link_libraries(${target} drawLib embeddingLib)
