set(HEADERS
        include/DrawCommon.hpp
        include/SVGDrawer.hpp
        include/IPEDrawer.hpp
)

set(SOURCES
        src/DrawCommon.cpp
        src/IPEDrawer.cpp
        src/SVGDrawer.cpp
)

add_library(drawLib STATIC ${SOURCES})

target_include_directories(drawLib PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(drawLib PUBLIC wembed_flags graphLib)
set_target_properties(drawLib PROPERTIES POSITION_INDEPENDENT_CODE TRUE)