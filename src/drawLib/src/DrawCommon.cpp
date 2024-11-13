#include "DrawCommon.hpp"

#include <cmath>

std::vector<std::vector<double>> Common::projectOntoPlane(const std::vector<std::vector<double>>& points) {
    std::vector<std::vector<double>> result(points.size(), std::vector<double>(2));

    if (points[0].size() >= 2) {
        for (int i = 0; i < points.size(); i++) {
            result[i][0] = points[i][0];
            result[i][1] = points[i][1];
        }
    } 
    // dimension equals 1
    else {
        for (int i = 0; i < points.size(); i++) {
            result[i][0] = points[i][0];
            result[i][1] = 0.0;
        }
    }
    return result;
}

// from: https://www.cs.rit.edu/~ncs/color/t_convert.html
Color Common::HSVtoRGB(float h, float s, float v) {
    int i;
    float f, p, q, t;
    float r, g, b;
    if (s == 0) {
        // achromatic (grey)
        return Color{v, v, v};
    }
    h *= (1.0 / 60.0);  // sector 0 to 5
    i = std::floor(h);
    f = h - i;  // factorial part of h
    p = v * (1 - s);
    q = v * (1 - s * f);
    t = v * (1 - s * (1 - f));
    switch (i) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        default:  // case 5:
            r = v;
            g = p;
            b = q;
            break;
    }
    return Color{255 * r, 255 * g, 255 * b};
}
