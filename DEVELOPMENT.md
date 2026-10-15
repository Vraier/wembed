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
The build configurations are defined in `CMakePresets.json`:

| Preset    | Purpose                                                          |
|-----------|------------------------------------------------------------------|
| `release` | optimized build                                                  |
| `debug`   | debug build with address and undefined behavior sanitizers       |
| `profile` | optimized build with debug info and frame pointers (perf/samply) |

Each preset builds into a folder with its name, which contains a `bin/` and `lib/` folder with the executables and libraries.
```
cmake --preset release
cmake --build --preset release
```

To rebuild after source changes, only the second command is needed.


## Running Tests

```
ctest --preset release
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


## Installing from TestPyPI

```
python -m venv /tmp/test-wembed-pypi
source /tmp/test-wembed-pypi/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ wembed
python python/examples/cli_example.py -i assets/small_graph.edg
```

`--extra-index-url` allows pip to fetch dependencies (e.g. pybind11) from the real PyPI when they are absent on TestPyPI.
