# WEmbed

This project contains the source code of `WEmbed` for calculating low dimensional weighted vertex embeddings. 
The library is written in C++ and includes Python bindings.
Below is an example of a two-dimensional embedding calculated by WEmbed.

![](https://raw.githubusercontent.com/Vraier/wembed/refs/heads/main/assets/internet_graph.jpg)
<sub> WEmbed embedding of the internet graph obtained from Boguñá, M., Papadopoulos, F. & Krioukov, D. Sustaining the Internet with hyperbolic mapping . Nat Commun 1, 62 (2010). https://doi.org/10.1038/ncomms1063 </sub>

The network represents the connections between internet routers.
Vertex size represents a weight, calculated by WEmbed, and
colors indicate the country of the respective routers IP-Address.
Note that WEmbed had no knowledge of the countries during the embedding process and still managed to assign vertices from the same countries similar spacial coordinates.


## Installing the Python module

On most Linux systems we provide prebuild binaries, and you should be able to install WEmbed via pip.
We recommend creating a new virtual environment before installing WEmbed.
```
python -m venv .venv
source .venv/bin/activate
pip install wembed
```
If your Linux system is not supported, or you are on Windows/Mac, pip will try to build WEmbed from source. 
In this case you have to make sure, that you install all necessary dependencies (see section further below).


## Usage and file formats

Both the [C++ example](https://github.com/Vraier/wembed/blob/main/src/cli_wembed/) and the [Python example](https://github.com/Vraier/wembed/blob/main/python/examples/cli_example.py) show how to use the code.
A minimal working example for the python bindings might look like this:

```
import wembed

graph = wembed.graphFromEdgeListFile("example.edg")
options = wembed.Options()
options.embeddingDimension = 4
embedder = wembed.createEmbedder(graph, options)

embedder.calculateEmbedding()

embedder.writeCoordinates("example.emb")
```

* Start by creating a graph object.
  This can be done with a file (`graphFromEdgeListFile`) or a list of `wembed.Edge` objects (`graphFromEdges`).
  The graph is assumed to be undirected, connected and with consecutive vertex ids starting at zero.
  The file is expected to contain one line per edge. Each edge should only be given in one direction.
  The repository contains a small [example graph file](https://github.com/Vraier/wembed/blob/main/assets/small_graph.edg).

* Create the embedder with `createEmbedder` from the `graph` object and an `options` object.
  You can modify the behavior of the embedder through this options object (e.g. changing the embedding dimension).
  You can calculate a single gradient descent step through `calculateStep()` or calculate until convergence with `calculateEmbedding()`.

* The final embedding can be written to file.
  It will contain one line per vertex.
  The first number of every line is the id of the vertex and the next d entries contain the coordinates for this vertex.
  The last entry represents the weight of the vertex.


## Installing Dependencies

In order to compile WEmbed you need a C++20 compiler with OpenMP support and CMake 3.28 or newer.
You can look at the [flake.nix](https://github.com/Vraier/wembed/blob/main/flake.nix) for more information.
WEmbed also depends on a few smaller libraries, these get downloaded automatically by CMake via Fetchcontent (you do not have to worry about them).

By default, WEmbed uses the [sprk tree](https://github.com/wembed-pdf/sprk) as its spatial index, which needs a recent Rust toolchain (`cargo` 1.88 or newer).
If you do not have Rust, configure with `-DWEMBED_USE_SPRK=OFF`: WEmbed then only contains a bundled KD-tree, which needs nothing but a C++ compiler.
The KD-tree is never chosen silently, you have to select it explicitly (`--index-type 0`, or `indexType = IndexKdTree` in the options); it is also the only index for more than 16 embedding dimensions.
Both indices report exactly the same repelling pairs, so the choice only affects the running time: expect the KD-tree to be a bit slower (x2.0) especially for datasets with few vertices or high dimensions.


## Compiling with CMake

The project uses CMake as a build tool (see the [root CMakeLists.txt](https://github.com/Vraier/wembed/blob/main/CMakeLists.txt) for more details).
In order to build the binaries clone this repository,
and call CMake with the `release` preset.
A `release` folder will be created, its `bin` and `lib` folders contain the executables and libraries.
```
git clone git@github.com:Vraier/wembed.git
cd wembed
cmake --preset release
cmake --build --preset release
```
This needs [Ninja](https://ninja-build.org/). Without it, `cmake -B release -DCMAKE_BUILD_TYPE=Release` followed by `cmake --build release -j4` does the same.

The build can be adjusted with the following options (e.g. `cmake --preset release -DWEMBED_USE_SPRK=OFF`):

| Option                | Default | Description                                                                                         |
|-----------------------|---------|-----------------------------------------------------------------------------------------------------|
| `WEMBED_USE_SPRK`     | `ON`    | Use the sprk tree as spatial index (needs Rust). If `OFF`, only the bundled KD-tree is available.    |
| `WEMBED_BUILD_TOOLS`  | `ON`    | Build the command line tools and unit tests. `OFF` by default if WEmbed is part of another project. |
| `WEMBED_BUILD_PYTHON` | `OFF`   | Build the python module (needs the Python development headers). The presets turn this on.           |

Run the unit tests with `ctest --preset release`.
See [DEVELOPMENT.md](https://github.com/Vraier/wembed/blob/main/DEVELOPMENT.md) for the other presets (debug with sanitizers, profiling) and for building wheels.


## Using WEmbed in your C++ project

The C++ interface is the single header [wembed.h](https://github.com/Vraier/wembed/blob/main/include/wembed.h) and the CMake target `wembed::wembed`.
With CMake's FetchContent, only the library is built, not the tools and tests:
```
include(FetchContent)
FetchContent_Declare(
    wembed
    GIT_REPOSITORY https://github.com/Vraier/wembed.git
    GIT_TAG        v0.2.0
)
FetchContent_MakeAvailable(wembed)

target_link_libraries(my_target PRIVATE wembed::wembed)
```


## Project Structure

All C++ source files can be found in [src](https://github.com/Vraier/wembed/blob/main/src/), this includes the [library](https://github.com/Vraier/wembed/blob/main/src/embeddingLib/) and small example command line applications for [C++](https://github.com/Vraier/wembed/blob/main/src/cli_wembed/). The [python](https://github.com/Vraier/wembed/blob/main/python/) folder contains code for the python bindings and an example using these bindings.
Unit tests using google test are found in [tests](https://github.com/Vraier/wembed/blob/main/tests/).


## Work in progress

Note that WEmbed is still quite experimental, expect major changes in the future. See [TODOs](https://github.com/Vraier/wembed/blob/main/TODOs.md) for pending TODOs.


