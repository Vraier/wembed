#pragma once

// from https://github.com/grybouilli/SFML-line-with-thickness

#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/Transformable.hpp>
#include <SFML/Graphics/VertexArray.hpp>
#include <SFML/Graphics/CircleShape.hpp>
#include <utility>
#include <vector>

static const float THICKNESS = 2.5;
static const float DOT_SIZE = 5.f;
static const sf::Color COLOR = sf::Color::White;

using point_set = std::vector<sf::Vector2f>;
class Thick_Line : public sf::Drawable, public sf::Transformable
{
public:
    Thick_Line();
    Thick_Line(const point_set &);

    void renew_shape(const point_set &);

    void add_point(sf::Vector2f);
    void pop_point();
    void close_line();

    void set_color(sf::Color);
    void set_thickness(float);

    float get_thickness() &&;
    const float &get_thickness() const &;

private:
    virtual void draw(sf::RenderTarget &target, sf::RenderStates states) const;

    void add_point_with_offset(sf::Vector2f, sf::Vector2f);
    std::pair<sf::Vector2f, sf::Vector2f> make_offset(sf::Vector2f, sf::Vector2f, sf::Vector2f);

    float _thickness;
    sf::Color _color;
    sf::VertexArray _shape;
    int _last_dot;
};

sf::Vector2f unit_vector(sf::Vector2f v);
sf::Vector2f mid_point(sf::Vector2f u, sf::Vector2f v);
sf::Vector2f normal_vector(sf::Vector2f u);
bool are_colinear(sf::Vector2f u, sf::Vector2f v);
sf::Vector2f intersection_point(sf::Vector2f dir1, sf::Vector2f pt1, sf::Vector2f dir2, sf::Vector2f pt2);
