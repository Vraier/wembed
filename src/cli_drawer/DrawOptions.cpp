#include "DrawOptions.hpp"

namespace po = boost::program_options;

DrawOptions::DrawOptions() : parser("Drawer") {
    OptionValues defaultOpt;

    parser.description.add_options()("help,h", po::bool_switch(&values.help)->default_value(false),
                                     "Prints an overview of all options.")  //
        ("edg-file,g", po::value<std::string>(&values.edgPath)->default_value(defaultOpt.edgPath),
         "Filepath to the edge list of a graph.")  //
        ("coord-file", po::value<std::string>(&values.coordFile)->default_value(defaultOpt.coordFile),
         "File with coordinates of the graph nodes.")  //
        ("w-coord-file,c", po::value<std::string>(&values.weightCoordFile)->default_value(defaultOpt.weightCoordFile),
         "File with coordinates and weights of the graph nodes.")  //
        ("svg-file,o", po::value<std::string>(&values.svgPath)->default_value(defaultOpt.svgPath),
         "Filepath to the SVG that contains the graph drawing.")  //
        ("ipe-file,p", po::value<std::string>(&values.ipePath)->default_value(defaultOpt.ipePath),
         "Filepath to the IPE-file that contains the graph drawing.");
}

bool DrawOptions::checkValidity() {
    if (values.help) return false;
    return true;
}
