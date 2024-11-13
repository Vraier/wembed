#pragma once

#include <vector>

using Coordinate = std::vector<double>;
using Coordinates = std::vector<Coordinate>;

/**
 * r,g,b between 0 and 255
 */
struct Color {
    float r;
    float g;
    float b;
};

class Common {
   public:
    static Coordinates projectOntoPlane(const Coordinates& points);
    /**
     * Value h is between 0 and 360
     */
    static Color HSVtoRGB(float h, float s, float v);
};