#pragma once

#include "GenOptions.hpp"
#include "Graph.hpp"

/**
 * this is basically a wrapper around the girg library.
 * if i understand everything correctly i can convince this library to output girgs and euclidean graphs for me.
 * even if they are not on the torus.
 */
class GirgGenerator {
   public:
    /**
     * Generates a girg with the given parameters.
     * Can write a file as a side effect containing the coordinates used during generation.
     */
    static Graph generateRandomGraph(OptionValues options);
};
