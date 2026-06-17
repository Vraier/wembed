include(FetchContent)

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
)
FetchContent_MakeAvailable(Corrosion)

FetchContent_Declare(
    sprk
    GIT_REPOSITORY https://github.com/wembed-pdf/sprk.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(sprk)

corrosion_import_crate(MANIFEST_PATH ${sprk_SOURCE_DIR}/Cargo.toml CRATES sprk-ffi CRATE_TYPES staticlib)
corrosion_set_env_vars(sprk_ffi "RUSTFLAGS=-C target-cpu=native")

set_target_properties(sprk_ffi-static PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${sprk_SOURCE_DIR}/sprk-ffi/include"
)
