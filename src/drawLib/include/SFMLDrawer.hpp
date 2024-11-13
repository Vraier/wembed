#pragma once

#include <SFML/Graphics.hpp>

#include "DrawCommon.hpp"
#include "Graph.hpp"

class SFMLDrawer {
   public:
    SFMLDrawer();
    ~SFMLDrawer();

    void processFrame(const Graph &g, const Coordinates &coords, const std::vector<Color> &colors);
    void processFrame(const Graph &g, const Coordinates &coords, const std::vector<double> &colScale);
    void processFrame(const Graph &g, const Coordinates &coords);

   private:
    void updateDisplay(const Graph &g, const Coordinates &coords, const std::vector<Color> &colors);
    void calculateBounds(const Coordinates &coords);

    double minX, minY, maxX, maxY;  // minimum/maximum coordinates of all vertices
    double width, height;

    sf::RenderWindow *window;
    sf::Clock *clock;

    static const int TARGETFRAMERATE = 60;
    int currentFrame = 0;
};