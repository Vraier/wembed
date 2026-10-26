# girgs declares its own options (shared libraries and its CLIs default to ON).
# CMP0077 lets the plain variables below override them without ending up in our cache.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(BUILD_SHARED_LIBS OFF)
set(OPTION_BUILD_CLI OFF)
FetchContent_Declare(
    girgs
    GIT_REPOSITORY https://github.com/chistopher/girgs.git
    GIT_TAG        d0e74e3f7ad714222a57d8cbea3e0f2fe5ec608b # master, 2023-10-23
    SYSTEM EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(girgs)

add_executable(generator main.cpp GirgGenerator.cpp)
target_link_libraries(generator PRIVATE wembed wembed_internal girgs::girgs CLI11::CLI11)
