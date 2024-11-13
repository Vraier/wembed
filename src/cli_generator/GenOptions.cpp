#include "GenOptions.hpp"

namespace po = boost::program_options;

GenOptions::GenOptions() : parser("Generator") {
    OptionValues defaultOpt;

    parser.description.add_options()
        // general settings
        ("seed,s", po::value<int>(&values.seed)->default_value(defaultOpt.seed),
         "Seed for random number generation. -1 Means using time for seed.")  //
        ("help,h", po::bool_switch(&values.help)->default_value(false),
         "Prints an overview of all options.")  //
        // graph output settings
        ("girg-file,o", po::value<std::string>(&values.girgFile)->default_value(defaultOpt.girgFile),
         "Filepath to generate a random girg graph to.")  //
        ("girg-coords", po::value<std::string>(&values.girgCoords)->default_value(defaultOpt.girgCoords),
         "Filepath to the coordinates of the generated gird. Each node gets on line. "
         "The first entry is the node id, then coordinates. The last entry is the weight")  //
        ("nodes,n", po::value<int>(&values.numNodes)->default_value(defaultOpt.numNodes),
         "Number of nodes of the unit disc euclidean graph.")  //
        ("ple", po::value<double>(&values.ple)->default_value(defaultOpt.ple),
         "Power law exponent for girg generation.")  //
        ("avg-deg", po::value<double>(&values.averageDegree)->default_value(defaultOpt.averageDegree),
         "Average degree of the generated graph.")  //
        ("gen-dim,d", po::value<int>(&values.genDimension)->default_value(defaultOpt.genDimension),
         "Dimension in which the graph is generated.")  //
        ("temperature", po::value<double>(&values.temperature)->default_value(defaultOpt.temperature),
         "Determins the randomness of edges. between 0 and 1")  //
        ("torus", po::bool_switch(&values.torus)->default_value(false),
         "If this flag is set, the generated graph is imbedded in the torus.");  //
}

bool GenOptions::checkValidity() {
    if (values.help) return false;
    return true;
}