set(target cli_generator)

add_executable(${target})

target_include_directories(${target} PRIVATE 
        ".")

target_sources(${target} PRIVATE
        "main.cpp"
        "GenOptions.cpp"
        "GirgGenerator.cpp")

target_link_libraries(${target} PRIVATE embeddingLib girgs::girgs)