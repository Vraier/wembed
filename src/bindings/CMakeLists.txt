# Add bindings for embedding lib
pybind11_add_module(_core bindings.cpp)
target_link_libraries(_core PRIVATE wembed_flags graphLib embeddingLib)
install(TARGETS _core LIBRARY DESTINATION wembed)
