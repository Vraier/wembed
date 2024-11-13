#include "SVGDrawer.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#include "Macros.hpp"

static const double VERTEXRADIUS = 0.15;
static const double EDGEWIDTH = 0.04;

SVGOutputWriter::SVGOutputWriter() : minX(0.0), minY(0.0), maxX(0.0), maxY(0.0){};

std::string SVGOutputWriter::drawText(std::string text, Coordinate p) {
    std::stringstream ss;
    ss << "<text x=\"" << p[0] << "\" y=\"" << p[1] << "\">" << text << "</text>";
    return ss.str();
}

std::string SVGOutputWriter::drawCircle(Coordinate p, double r) {
    Color c{0, 0, 0};
    return drawCircle(p, r, c);
}

std::string SVGOutputWriter::drawCircle(Coordinate p, double r, Color c) {
    std::stringstream ss;
    ss << "<circle cx=\"" << p[0] << "\" cy=\"" << p[1] << "\" r=\"" << r << "\" style=\"fill:rgb(" << c.r << "," << c.g
       << "," << c.b << ")\"/>";
    return ss.str();
}

std::string SVGOutputWriter::drawLine(Coordinate p1, Coordinate p2, double w) {
    Color c{0, 0, 0};
    return drawLine(p1, p2, w, c);
}

std::string SVGOutputWriter::drawLine(Coordinate p1, Coordinate p2, double w, Color c) {
    std::stringstream ss;
    ss << "<line x1=\"" << p1[0] << "\" y1=\"" << p1[1] << "\" x2=\"" << p2[0] << "\" y2=\"" << p2[1]
       << "\" style=\"stroke:rgb(" << c.r << "," << c.g << "," << c.b << ");stroke-width:" << w << "\"/>";
    return ss.str();
}

std::string SVGOutputWriter::svgBeginning() {
    std::stringstream ss;
    ss << "<svg version='1.1' baseProfile='full' xmlns='http://www.w3.org/2000/svg' width=\"100%\" viewBox=\""
       << minX << " " << minY << " " << width << " " << height
       << "\" style=\"background-color:gray\">\n"
       << "<rect x='" << minX << "' y='" << minY << "' width=\" " << width << " \" height=\" " << height << " \" fill=\"white\"/>";
    return ss.str();
}

std::string SVGOutputWriter::svgEnd() {
    std::stringstream ss;
    ss << "</svg>";
    return ss.str();
}

void SVGOutputWriter::calculateBounds(const Coordinates &coords) {
    int n = coords.size();

    ASSERT(n > 0);

    maxX = coords[0][0];
    minX = coords[0][0];
    maxY = coords[0][1];
    minY = coords[0][1];
    for (int i = 0; i < n; i++) {
        maxX = std::max(maxX, coords[i][0]);
        minX = std::min(minX, coords[i][0]);
        maxY = std::max(maxY, coords[i][1]);
        minY = std::min(minY, coords[i][1]);
    }

    width = maxX - minX;
    height = maxY - minY;

    double padding = std::min(width, height) * 0.05;
    minX -= padding;
    minY -= padding;
    maxX += padding;
    maxY += padding;

    width = maxX - minX;
    height = maxY - minY;
}

void SVGOutputWriter::write(const std::string &path, const Graph &graph, const Coordinates &coords, const std::vector<Color> &colors) {
    // open file
    std::ofstream file;
    file.open(path);

    // calculate bounds
    calculateBounds(coords);
    int n = graph.getNumVertices();

    // calculate node size dynamically
    double viewArea = width * height;
    double perNodeAreaLength = std::sqrt(viewArea / n);
    double nodeRadius = 0.1 * perNodeAreaLength;
    double edgeWidth = 0.3 * nodeRadius;

    // prefix
    file << svgBeginning() << "\n";

    // draw edges
    for (int v = 0; v < n; ++v) {
        std::vector<int> neighbors = graph.getNeighbors(v);
        for (int u : neighbors) {
            Coordinate posU = coords[u];
            Coordinate posV = coords[v];
            if (v > u)  // only draw each edge once
                file << drawLine(posU, posV, edgeWidth) << "\n";
        }
    }

    // draw vertices, Over Edges!
    for (int v = 0; v < n; ++v) {
        Coordinate pos = coords[v];
        Color col = colors[v];
        file << drawCircle(pos, nodeRadius, col) << "\n";
    }

    // suffix
    file << svgEnd();

    // close file
    file.close();
}

void SVGOutputWriter::write(const std::string &path, const Graph &graph, const Coordinates &coords, const std::vector<double> &colScale) {
    std::vector<Color> colors(graph.getNumVertices());
    // scale wweights to be between 0 and one
    double maxNodeWeight = 0;
    std::vector<double> scaledWeights(colScale.size());
    for (NodeId v = 0; v < colScale.size(); v++) {
        maxNodeWeight = std::max(maxNodeWeight, colScale[v]);
    }
    for (NodeId v = 0; v < colScale.size(); v++) {
        scaledWeights[v] = colScale[v] / maxNodeWeight;
    }
    for (int i = 0; i < graph.getNumVertices(); i++) {
        colors[i] = Common::HSVtoRGB(scaledWeights[i] * 360, 1, 1);  // scale h to be between 0 and 60
    }
    write(path, graph, coords, colors);
}

void SVGOutputWriter::write(const std::string &path, const Graph &graph, const Coordinates &coords) {
    std::vector<Color> colors(graph.getNumVertices());
    for (int i = 0; i < graph.getNumVertices(); i++) {
        colors[i] = Color{100, 100, 100};  // gray default color for nodes
    }
    write(path, graph, coords, colors);
}
