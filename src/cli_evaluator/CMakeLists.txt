set(target evaluator)

add_executable(${target})

target_include_directories(${target} PRIVATE ".")

target_sources(${target} PRIVATE "main.cpp")

target_link_libraries(${target} PRIVATE evaluationLib CLI11::CLI11)