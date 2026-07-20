set(target wembed)

add_executable(${target})

target_include_directories(${target} PRIVATE ".")

target_sources(${target} PRIVATE
        "main.cpp")

target_link_libraries(${target} PRIVATE wembed_flags wembedLib CLI11::CLI11)