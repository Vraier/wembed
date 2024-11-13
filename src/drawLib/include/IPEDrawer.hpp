#pragma once

#include "DrawCommon.hpp"
#include "Graph.hpp"

#include <fstream>
#include <string>

/**
 * Output Writer in ipe Format.
 * **/

class IpeOutputWriter {
   public:
    IpeOutputWriter(const std::string& path, double scaling = 1.0, double margin = 0.0);
    ~IpeOutputWriter();

    void write_graph(const Graph& graph, const Coordinates& coords);
    void write_graph(const Graph& graph, const Coordinates& coords, const std::vector<double>& color_scale);
    void write_graph(const Graph& graph, const Coordinates& coords, const std::vector<Color> & colors);

   private:
    std::ofstream m_file;

    void file_start();
    void file_end();

    void page_start();
    void page_end();

    double m_scaling;
    double m_margin;
    void normalize_point(double& x, double& y);

    void new_page();

    void label(const std::string& label, double x, double y, const std::string& color = "black");
    void line(double x1, double y1, double x2, double y2, const std::string& color = "gray2",
              const std::string& additional_settings = "");
    void box(double x1, double y1, double x2, double y2, const std::string& color = "black");
    void point(double x, double y, const std::string& color = "black");
    void point(double x, double y, Color color);
    void disk(double x, double y, double radius, const std::string& color = "black");

    void start_group();
    void start_group_with_clipping(double x1, double y1, double x2, double y2);
    void end_group();
};
