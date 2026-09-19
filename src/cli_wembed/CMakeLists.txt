# the target cannot be called wembed, that is the library
add_executable(wembed_cli main.cpp)
set_target_properties(wembed_cli PROPERTIES OUTPUT_NAME wembed)
target_link_libraries(wembed_cli PRIVATE wembed_flags wembed CLI11::CLI11)
