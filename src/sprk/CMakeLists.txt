FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG        v0.5.2
    GIT_SHALLOW    ON
)
FetchContent_Declare(
    sprk
    GIT_REPOSITORY https://github.com/wembed-pdf/sprk.git
    GIT_TAG        1e195eab1119aa5cf420bfb322788d53b95bf1ad # main, 2026-04-07
)
FetchContent_MakeAvailable(Corrosion sprk)

corrosion_import_crate(MANIFEST_PATH ${sprk_SOURCE_DIR}/Cargo.toml CRATES sprk-ffi CRATE_TYPES staticlib)
# wheels have to run on other machines than the one they are built on
if(NOT SKBUILD)
    corrosion_set_env_vars(sprk_ffi "RUSTFLAGS=-C target-cpu=native")
endif()

set_target_properties(sprk_ffi-static PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${sprk_SOURCE_DIR}/sprk-ffi/include"
)
