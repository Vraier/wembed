include(FetchContent)

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
)
FetchContent_MakeAvailable(Corrosion)

FetchContent_Declare(
    atree
    GIT_REPOSITORY https://github.com/wembed-pdf/atree.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(atree)

corrosion_import_crate(MANIFEST_PATH ${atree_SOURCE_DIR}/Cargo.toml CRATES atree-ffi)
corrosion_set_env_vars(atree_ffi "RUSTFLAGS=-C target-cpu=native")

# Expose the C header for consumers via the corrosion-generated target (atree_ffi)
set_target_properties(atree_ffi PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${atree_SOURCE_DIR}/atree-ffi/include"
)
