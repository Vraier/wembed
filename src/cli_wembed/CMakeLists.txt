set(target wembed)

add_executable(${target})

target_include_directories(${target} PRIVATE ".")

target_sources(${target} PRIVATE
        "main.cpp")

if(SFML_FOUND)
        target_compile_definitions(${target} PRIVATE "EMBEDDING_USE_ANIMATION")
endif()

target_link_libraries(${target} PRIVATE wembed_flags embeddingLib CLI11::CLI11 drawLib OpenMP::OpenMP_CXX)