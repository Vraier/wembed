set(target drawLib)

set(HEADERS
        include/DrawCommon.hpp
        include/SVGDrawer.hpp
        include/SFMLDrawer.hpp
        include/IPEDrawer.hpp
        include/ThickLine.hpp
)

set(SOURCES
        src/DrawCommon.cpp
        src/SFMLDrawer.cpp
        src/IPEDrawer.cpp
        src/ThickLine.cpp
        src/SVGDrawer.cpp
)

add_library(${target} SHARED ${HEADERS} ${SOURCES})

target_include_directories(${target} PUBLIC 
        ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(${target} PUBLIC graphLib sfml-graphics)
