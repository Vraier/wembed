#include "SFMLDrawer.hpp"

#include <cmath>

#include "Macros.hpp"
#include "ThickLine.hpp"

SFMLDrawer::SFMLDrawer() {
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    window = new sf::RenderWindow(sf::VideoMode(1600, 900), "2D Graph Simulation",
                                  sf::Style::Titlebar | sf::Style::Close, settings);
    clock = new sf::Clock();

    window->clear();
    window->display();
    window->setFramerateLimit(TARGETFRAMERATE);
}

SFMLDrawer::~SFMLDrawer() {
    window->close();
    delete window;
    delete clock;
}

void SFMLDrawer::processFrame(const Graph &g, const Coordinates &coords, const std::vector<Color> &colors) {
    // handle events
    sf::Event event;
    while (window->pollEvent(event)) {
        if (event.type == sf::Event::Closed) window->close();
    }
    // draw to screen
    updateDisplay(g, coords, colors);
    currentFrame++;
}

void SFMLDrawer::processFrame(const Graph &g, const Coordinates &coords, const std::vector<double> &colScale) {
    ASSERT(g.getNumVertices() == colScale.size());
    ASSERT(g.getNumVertices() == coords.size());

    std::vector<Color> colors(g.getNumVertices());
    // scale weights to be between 0 and one
    double maxNodeWeight = 0;
    std::vector<double> scaledWeights(colScale.size());
    for (NodeId v = 0; v < colScale.size(); v++) {
        maxNodeWeight = std::max(maxNodeWeight, colScale[v]);
    }
    for (NodeId v = 0; v < colScale.size(); v++) {
        scaledWeights[v] = colScale[v] / maxNodeWeight;
    }
    for (int i = 0; i < g.getNumVertices(); i++) {
        colors[i] = Common::HSVtoRGB(scaledWeights[i] * 360, 1, 1);  // scale h to be between 0 and 60
    }
    processFrame(g, coords, colors);
}

void SFMLDrawer::processFrame(const Graph &g, const Coordinates &coords) {
    std::vector<Color> colors(g.getNumVertices());
    for (int i = 0; i < g.getNumVertices(); i++) {
        colors[i] = Color{100, 100, 100};  // gray default color for nodes
    }
    processFrame(g, coords, colors);
}

void SFMLDrawer::calculateBounds(const Coordinates &coords) {
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

void SFMLDrawer::updateDisplay(const Graph &g, const Coordinates &coords, const std::vector<Color> &colors) {
    ASSERT(!coords.empty());
    ASSERT(coords[0].size() == 2);
    window->clear(sf::Color(220, 220, 220, 255));

    if (!window->isOpen()) {
        return;
    }

    // determine which axis has to get scaled to keep aspect ratio
    calculateBounds(coords);
    double currRatio = width / height;
    if (currRatio < 16.0 / 9.0) {
        width = 16.0 / 9.0 * height;
    } else {
        height = 9.0 / 16.0 * width;
    }

    // set the view
    sf::View view(sf::FloatRect(minX, minY, width, height));
    window->setView(view);

    // calculate node size dynamically
    double viewArea = width * height;
    double perNodeAreaLength = std::sqrt(viewArea / g.getNumVertices());
    double nodeRadius = 0.1 * perNodeAreaLength;
    double edgeWidth = 0.3 * nodeRadius;

    // draw the edges
    int n = g.getNumVertices();
    for (int v = 0; v < n; v++) {
        for (int u : g.getNeighbors(v)) {
            Thick_Line l;
            l.add_point(sf::Vector2f(coords[v][0], coords[v][1]));
            l.add_point(sf::Vector2f(coords[u][0], coords[u][1]));
            l.set_thickness(edgeWidth);
            l.set_color(sf::Color::Black);
            window->draw(l);
        }
    }

    // draw the nodes
    for (int v = 0; v < n; v++) {
        sf::CircleShape shape(nodeRadius);
        shape.setFillColor(sf::Color(colors[v].r, colors[v].g, colors[v].b));

        shape.setOrigin(sf::Vector2f(nodeRadius, nodeRadius));
        shape.setPosition(sf::Vector2f(coords[v][0], coords[v][1]));
        window->draw(shape);
    }

    window->display();
}