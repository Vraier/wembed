set(target generator)

add_executable(${target})

target_include_directories(${target} PRIVATE 
        ".")

target_sources(${target} PRIVATE
        "main.cpp"
        "GirgGenerator.cpp")

target_link_libraries(${target} PRIVATE wembed_flags embeddingLib girgs::girgs CLI11::CLI11)