FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v3.0.4
    GIT_SHALLOW    ON
    SYSTEM EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(wembed_python bindings.cpp)
target_link_libraries(wembed_python PRIVATE wembed_flags wembed)
set_target_properties(wembed_python PROPERTIES OUTPUT_NAME wembed)

install(TARGETS wembed_python LIBRARY DESTINATION . COMPONENT python)
