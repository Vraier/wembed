{
  description = "WEmbed - Calculate low dimensional weighted node embeddings";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # Sources that CMake would otherwise download (keep in sync with the pins in the CMake files)
    sprk = {
      url = "github:wembed-pdf/sprk/1e195eab1119aa5cf420bfb322788d53b95bf1ad";
      flake = false;
    };
    girgs = {
      url = "github:chistopher/girgs/d0e74e3f7ad714222a57d8cbea3e0f2fe5ec608b";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    sprk,
    girgs,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "wembed";
          version = "0.2.0";
          src = ./.;

          nativeBuildInputs = with pkgs; [cmake ninja cargo rustc];

          # The build sandbox has no network: hand CMake and cargo everything they would download.
          cmakeFlags = [
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
            "-DFETCHCONTENT_SOURCE_DIR_CORROSION=${pkgs.corrosion.src}"
            "-DFETCHCONTENT_SOURCE_DIR_SPRK=${sprk}"
            "-DFETCHCONTENT_SOURCE_DIR_CLI11=${pkgs.cli11.src}"
            "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${pkgs.gtest.src}"
            "-DFETCHCONTENT_SOURCE_DIR_GIRGS=${girgs}"
          ];
          preConfigure = ''
            export CARGO_HOME=$(mktemp -d)
            cat > $CARGO_HOME/config.toml <<EOF
[source.crates-io]
replace-with = "vendored-sources"
[source.vendored-sources]
directory = "${pkgs.rustPlatform.importCargoLock {lockFile = "${sprk}/Cargo.lock";}}"
[net]
offline = true
EOF
          '';

          doCheck = true;
          installPhase = "install -Dm755 -t $out/bin bin/wembed bin/evaluator bin/generator";

          meta = with pkgs.lib; {
            description = "Calculate low dimensional weighted node embeddings";
            homepage = "https://github.com/Vraier/wembed";
            license = {
              fullName = "MIT License";
              url = "https://opensource.org/licenses/MIT";
              spdxId = "MIT";
              file = ./LICENSE;
            };
            platforms = platforms.linux;
            maintainers = [
              {
                name = "Jean-Pierre von der Heydt";
                email = "heydt@kit.edu";
                github = "Vraier";
              }
              {
                name = "Nikolai Maas";
                email = "nikolai.maas@kit.edu";
              }
              {
                name = "Dennis Kobert";
                email = "dennis@kobert.dev";
                github = "TrueDoctor";
              }
            ];
          };
        };

        # Add apps to make the CLIs directly runnable
        apps.default = flake-utils.lib.mkApp {
          drv = self.packages.${system}.default;
          name = "wembed";
        };

        devShells.default = pkgs.mkShell {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];

          packages = with pkgs; [
            # Build tools
            cmake
            ninja
            pkg-config
            git

            # Rust
            cargo
            rustc
            rustfmt
            clippy

            # Core dependencies
            gtest

            # Python tools and dependencies
            python3
            python3.pkgs.scikit-build-core
            python3.pkgs.pybind11
            python3.pkgs.pip
            python3.pkgs.virtualenv
            pipx # for wheel building
            podman # for wheel building

            # Development tools
            gdb
            valgrind
            samply
            ccache
            clang-tools # For clang-format, clang-tidy
            pre-commit

            # Additional Python development tools
            python3.pkgs.pytest
            python3.pkgs.black
            python3.pkgs.flake8
          ];

          shellHook = ''
            echo "Welcome to WEmbed development environment!"
            echo "Build tools and dependencies are available."

            # Setup ccache
            export CCACHE_DIR=$PWD/.ccache
            export PATH="${pkgs.ccache}/bin:$PATH"

            # Make tests verbose by default
            export CTEST_OUTPUT_ON_FAILURE=1

            # Use podman for cibuildwheel (NixOS has no Docker daemon)
            export CIBW_CONTAINER_ENGINE=podman

            # Create virtual environment if it doesn't exist
            if [ ! -d ".venv" ]; then
              python -m venv .venv
              source .venv/bin/activate
              pip install -e .
            else
              source .venv/bin/activate
            fi
          '';
        };
      }
    );
}
