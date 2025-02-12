# WEmbed

This project contains the source code of `WEmbed` for calculating low dimensional weighted node embeddings. 
The library is written in C++ and includes Python bindings.
Below is an example of a two-dimensional embedding of [1] calculated by WEmbed.

![](assets/internet_graph.jpg)

This network represents the connection between internet routers.
Node size represents weight calculated by WEmbed and
colors indicate the country of the IP-Address reported by the respective router.
Note that WEmbed had no knowledge of the countries during the embedding process and still managed to assign nodes from the same countries similar spacial coordinates.

[1] Boguñá, M., Papadopoulos, F. & Krioukov, D. Sustaining the Internet with hyperbolic mapping . Nat Commun 1, 62 (2010). https://doi.org/10.1038/ncomms1063


## Installing the Python module

On most Linux systems we provide prebuild binaries, and you should be able to install WEmbed via pip.
We recommend creating a new virtual environment before installing WEmbed.
```
python -m venv .venv
source .venv/bin/activate
pip install wembed
```
If your Linux system is not supported, or you are on Windows/Mac, pip will try to build WEmbed from source. 
In this case you have to make sure, that you install all necessary dependencies.


## Installing Dependencies

In order to compile WEmbed you need to have `Eigen3` and `Boost` headers installed.
You can look at the development [Dockerfile](docker_dev/Dockerfile) for more information.
WEmbed also depends on a few other smaller libraries, these get downloaded automatically by CMake via Fetchcontent (so you do not have to worry about them), 
look at the root [CMakeLists.txt](CMakeLists.txt) for more information.


## Compiling with CMake

The project uses CMake as a build tool (see the [root CMakeLists.txt](CMakeLists.txt) for more details).
In order to build the binaries clone this repository,
create a new folder and call CMake from it.
A `bin` and `lib` folder will be created containing the executables and libraries.
```
git clone git@github.com:Vraier/wembed.git
cd wembed
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```


## Structure

All source files can be found in [src](src/), this includes the [library](src/embeddingLib/), [python bindings](src/bindings/) and small example command line applications for [C++](src/cli_wembed/) and [Python](src/cli_python_example/).
Unit tests using google test are found in [tests](tests/).
If you want to run WEmbed in a docker container, you can use the Dockerfile in [docker_dev](docker_dev/) ([docker_build](docker_build/) is used to build python packages).


## Usage and file formats

Both the [C++ example](src/cli_wembed/) and the [Python example](src/cli_python_example/) show how to use the code.

* Start by creating a graph object.
  This can be done with a file or a `vector of pairs` representing an edge list.
  The graph is assumed to be undirected, connected and with consecutive node ids starting at zero.
  The file is expected to contain one line per edge and have to be given in only one direction.
  The repository contains a small [example graph file](assets/small_graph.edg).

* Initialize the embedder with the `graph` object and an `options` object.
  You can modify the behavior of the embedder through this options object (e.g. changing the embedding dimension).
  You can calculate a single gradient descent step through `calculateStep()` or calculate until convergence with `calculateEmbedding()`

* The final embedding can be written to file.
  It will contain one line per node.
  The first number of every line is the id of the node and the next d entries contain the coordinates of this node.
  The last entry represents the weight of the node


## Work in progress

Note that WEmbed is still quiet experimental, expect major changes in the future. Some code sections that will be changed in the immediate future include:

* Make a larger portion of the library accessible through the Python bindings
* The repository contains some embedding code that is dead or outdated. This has to be updated or removed
