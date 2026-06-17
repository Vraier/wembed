# Development

## Prerequisites

The development environment is managed with [Nix](https://nixos.org/) and [direnv](https://direnv.net/).
After installing both, create an `.envrc` file and run `direnv allow` once in the repository root to activate the shell automatically.
This provides all build tools, compilers, and Python dependencies.

```
echo "use flake" > .envrc
direnv allow
```

Building wheels locally also requires Podman, which needs to be enabled in the NixOS system configuration:
```nix
virtualisation.podman.enable = true;
```


## Building

The project uses CMake (with Ninja, which automatically uses all available CPU cores).
A `bin/` and `lib/` folder will be created inside the build directory containing the executables and libraries.
```
mkdir release
cd release
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja
```

To rebuild after source changes:
```
cd release
ninja
```


## Running Tests

```
cd release
ctest
```


## Testing the Python Script

The Nix shell hook creates a `.venv` and runs `pip install -e .` on first use.
Activate it, then run the example script against the included sample graph:
```
source .venv/bin/activate
python python/examples/cli_example.py -i assets/small_graph.edg
```


## Building a Wheel Locally

This replicates the CI wheel build on your machine using the same manylinux container.
The resulting wheel is placed in `wheelhouse/`.
```
pipx run cibuildwheel --only cp313-manylinux_x86_64
```

To test the built wheel in a clean environment:
```
python -m venv /tmp/test-wembed
/tmp/test-wembed/bin/pip install wheelhouse/*.whl
/tmp/test-wembed/bin/python python/examples/cli_example.py -i assets/small_graph.edg
```
