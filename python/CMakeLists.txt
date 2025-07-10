FetchContent_Declare(
    pybind11 SYSTEM EXCLUDE_FROM_ALL
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(wembed_python bindings.cpp)

target_link_libraries(wembed_python PRIVATE wembed_flags graphLib embeddingLib)
set_target_properties(wembed_python PROPERTIES OUTPUT_NAME wembed)

install(TARGETS wembed_python LIBRARY DESTINATION . COMPONENT python)