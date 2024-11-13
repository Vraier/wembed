#pragma once

#include "DrawCommon.hpp"
#include "Graph.hpp"

/**
 * Output Writer in SVG Format.
 * **/

class SVGOutputWriter {

   public:
    SVGOutputWriter();

    void write(const std::string &path, const Graph &graph, const Coordinates &coords, const std::vector<Color> &colors);
    /**
     * colScale should have values between 0 and 1.
     * They get mapped to HSV space
     */
    void write(const std::string &path, const Graph &graph, const Coordinates &coords, const std::vector<double> &colScale);
    void write(const std::string &path, const Graph &graph, const Coordinates &coords);

   private:
    double minX, minY, maxX, maxY;  // minimum/maximum coordinates of all vertices
    double width, height;

    void calculateBounds(const Coordinates &coords);

    std::string drawText(std::string text, Coordinate p);
    std::string drawCircle(Coordinate p, double r);
    std::string drawCircle(Coordinate p, double r, Color c);
    std::string drawLine(Coordinate p1, Coordinate p2, double w);
    std::string drawLine(Coordinate p1, Coordinate p2, double w, Color c);

    std::string svgBeginning();
    std::string svgEnd();
};
