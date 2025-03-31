{
  description = "WEmbed - Calculate low dimensional weighted node embeddings";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # CMake dependencies
    googletest = {
      url = "github:google/googletest/release-1.12.1";
      flake = false;
    };
    cli11 = {
      url = "github:CLIUtils/CLI11/v2.3.2";
      flake = false;
    };
    girgs = {
      url = "github:chistopher/girgs/master";
      flake = false;
    };
    pybind11 = {
      url = "github:pybind/pybind11/v2.13.6";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, googletest, cli11, girgs, pybind11 }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "wembed";
          version = "0.0.1";

          src = ./.;

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
            python3
            git
            python3.pkgs.scikit-build-core
            python3.pkgs.pybind11
          ];

          buildInputs = with pkgs; [
            eigen
            boost
            gtest
            sfml
          ];

          # Copy the pre-fetched dependencies to their expected locations
          preConfigure = ''
            mkdir -p build/_deps
            cp -r ${googletest} build/_deps/googletest-src
            cp -r ${cli11} build/_deps/cli11-src
            cp -r ${girgs} build/_deps/girgs-src
            cp -r ${pybind11} build/_deps/pybind11-src
            chmod -R +w build/_deps
          '';

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
            "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${googletest}"
            "-DFETCHCONTENT_SOURCE_DIR_CLI11=${cli11}"
            "-DFETCHCONTENT_SOURCE_DIR_GIRGS=${girgs}"
            "-DFETCHCONTENT_SOURCE_DIR_PYBIND11=${pybind11}"
          ];

          # Handle Python packaging
          postInstall = ''
            # Ensure Python package is installed correctly
            export PYTHONPATH="$out/${python.sitePackages}:$PYTHONPATH"
            
            # Make the CLI executable available
            mkdir -p $out/bin
            cp bin/cli_wembed $out/bin/wembed
            chmod +x $out/bin/wembed
          '';

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
            maintainers = with maintainers; [
              (maintainers.lib.maintainer {
                name = "Jean-Pierre von der Heydt";
                email = "heydt@kit.edu";
                github = "Vraier";
              })
              (maintainers.lib.maintainer {
                name = "Nikolai Maas";
                email = "nikolai.maas@kit.edu";
              })
              (maintainers.lib.maintainer {
                name = "Dennis Kobert";
                email = "dennis@kobert.dev";
                github = "TrueDoctor";
              })
            ];
          };
        };

        # Add apps to make the CLIs directly runnable
        apps.default = flake-utils.lib.mkApp {
          drv = self.packages.${system}.default;
          name = "wembed";
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # Build tools
            cmake
            ninja
            pkg-config
            git
            
            # Core dependencies
            eigen
            boost
            gtest
            sfml
            
            # Python tools and dependencies
            python3
            python3.pkgs.scikit-build-core
            python3.pkgs.pybind11
            python3.pkgs.pip
            python3.pkgs.virtualenv
            
            # Development tools
            gdb
            valgrind
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

            # Setup Python environment variables
            export PYTHONPATH="$PWD:$PYTHONPATH"
            
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
