set(target cli_test)

add_executable(${target})

target_include_directories(${target} PRIVATE ".")

target_sources(${target} PRIVATE
        "main.cpp")

target_link_libraries(${target} PRIVATE embeddingLib CLI11::CLI11)