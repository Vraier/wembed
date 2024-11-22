set(target utilLib)

set(HEADERS
        include/FileOperations.hpp
        include/Macros.hpp 
        include/Rand.hpp
        include/StringManipulation.hpp
        include/Toolkit.hpp
        include/Timings.hpp
)

set(SOURCES
        src/FileOperations.cpp
        src/Rand.cpp
        src/StringManipulation.cpp
        src/Toolkit.cpp
        src/Timings.cpp
)

add_library(${target} STATIC ${HEADERS} ${SOURCES})

target_include_directories(${target} PUBLIC 
        ${CMAKE_CURRENT_LIST_DIR}/include)

set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(${target} PUBLIC wembed_flags)