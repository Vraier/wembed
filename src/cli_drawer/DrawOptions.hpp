#pragma once

#include "OptionParser.hpp"

struct OptionValues {
    bool help;

    std::string edgPath = "";

    std::string coordFile = "";
    std::string weightCoordFile = "";

    std::string svgPath = "";
    std::string ipePath = "";
};

class DrawOptions {
   public:
    OptionValues values;
    OptionParser parser;
    DrawOptions();

    bool checkValidity();
};
