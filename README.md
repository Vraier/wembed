# WEmbed

This project contains the source code of `WEmbed` for calculating low dimensional weighted vertex embeddings. 
The library is written in C++ and includes Python bindings.
Below is an example of a two-dimensional embedding calculated by WEmbed.

![](https://raw.githubusercontent.com/Vraier/wembed/refs/heads/main/assets/internet_graph.jpg)
<sub> WEmbed embedding of the internet graph by obtained from Boguñá, M., Papadopoulos, F. & Krioukov, D. Sustaining the Internet with hyperbolic mapping . Nat Commun 1, 62 (2010). https://doi.org/10.1038/ncomms1063 </sub>

This network represents the connection between internet routers.
Vertex size represents weight calculated by WEmbed and
colors indicate the country of the IP-Address of the respective router.
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

graph = wembed.readEdgeList("example.edg")
emb_opt = wembed.EmbedderOptions()
emb = wembed.Embedder(graph, emb_opt)

emb.calculateEmbedding()

wembed.writeCoordinates("example.emb", emb.getCoordinates(), emb.getWeights())
```

* Start by creating a graph object.
  This can be done with a file or a `vector of pairs` representing an edge list.
  The graph is assumed to be undirected, connected and with consecutive vertex ids starting at zero.
  The file is expected to contain one line per edge. Each edge should only be given in one direction.
  The repository contains a small [example graph file](https://github.com/Vraier/wembed/blob/main/assets/small_graph.edg).

* Initialize the embedder with the `graph` object and an `options` object.
  You can modify the behavior of the embedder through this options object (e.g. changing the embedding dimension).
  You can calculate a single gradient descent step through `calculateStep()` or calculate until convergence with `calculateEmbedding()`.

* The final embedding can be written to file.
  It will contain one line per vertex.
  The first number of every line is the id of the vertex and the next d entries contain the coordinates for this vertex.
  The last entry represents the weight of the vertex.


## Installing Dependencies

In order to compile WEmbed you need to have `Eigen3` and `Boost` headers installed.
You can look at the development [Dockerfile](https://github.com/Vraier/wembed/blob/main/docker_dev/Dockerfile) for more information.
WEmbed also depends on a few other smaller libraries, these get downloaded automatically by CMake via Fetchcontent (you do not have to worry about them), 
look at the root [CMakeLists.txt](https://github.com/Vraier/wembed/blob/main/CMakeLists.txt) for more information.


## Compiling with CMake

The project uses CMake as a build tool (see the [root CMakeLists.txt](https://github.com/Vraier/wembed/blob/main/CMakeLists.txt) for more details).
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


## Project Structure

All C++ source files can be found in [src](https://github.com/Vraier/wembed/blob/main/src/), this includes the [library](https://github.com/Vraier/wembed/blob/main/src/embeddingLib/) and small example command line applications for [C++](https://github.com/Vraier/wembed/blob/main/src/cli_wembed/). The [python](https://github.com/Vraier/wembed/blob/main/python/) folder contains code for the python bindings and an example using these bindings.
Unit tests using google test are found in [tests](https://github.com/Vraier/wembed/blob/main/tests/).
If you want to run WEmbed in a docker container, you can use the Dockerfile in [docker_dev](https://github.com/Vraier/wembed/blob/main/docker_dev/) ([docker_build](https://github.com/Vraier/wembed/blob/main/docker_build/) is used to build python packages).


## Work in progress

Note that WEmbed is still quite experimental, expect major changes in the future. Some code sections that will be changed in the immediate future include:

* The repository contains some embedding code that is dead or outdated. This has to be updated or removed
