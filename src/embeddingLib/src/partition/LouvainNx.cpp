#include "LouvainNx.hpp"

#include <fstream>

#include "GraphIO.hpp"

NodeParentPointer LouvainNxPartitioner::coarsenAllLayers(OptionValues options, const EmbeddedGraph& g) {
    unused(options);

    std::string adj_seed = std::to_string(Rand::randomInt(0, 10000000));
    std::string tmp_adj_file = "./adjList_" + adj_seed + ".txt";
    std::string tmp_tree_file = "./tree_" + adj_seed + ".txt";
    GraphIO::writeToAdjList(tmp_adj_file, g.graph);
    // TODO(JP) think about how to handle paths here
    std::string call = "python3 /home/jeanp/Documents/graphembedding/python/networkX.py -i " + tmp_adj_file + " -o " +
                       tmp_tree_file + " -s " + adj_seed;
    LOG_INFO( "Calling networkX via: " << call);
    int ret = system(call.c_str());
    unused(ret);

    return readInFileData(tmp_tree_file);
}

NodeParentPointer LouvainNxPartitioner::readInFileData(std::string filePath) {
    LOG_INFO( "Beginning to read parent pointer");

    std::ifstream input(filePath);
    std::string line;
    NodeParentPointer result;
    while (getline(input, line)) {
        std::istringstream is(line);
        result.push_back(std::vector<int>(std::istream_iterator<int>(is), std::istream_iterator<int>()));
    }
    LOG_INFO( "Finished reading data. Obtained " << result.size() << " layers");
    return result;
}
