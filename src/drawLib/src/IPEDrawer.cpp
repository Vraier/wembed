#include "IPEDrawer.hpp"

#include <iomanip>

IpeOutputWriter::IpeOutputWriter(const std::string& filename, double scaling_factor, double margin)
    : m_file(filename), m_scaling(scaling_factor), m_margin(margin) {
    m_file << std::fixed << std::setprecision(1);
    file_start();
    page_start();
}

IpeOutputWriter::~IpeOutputWriter() {
    page_end();
    file_end();
}

void IpeOutputWriter::write_graph(const Graph& graph, const Coordinates& coords) {
    std::vector<Color> colors(graph.getNumVertices());
    for (int i = 0; i < graph.getNumVertices(); i++) {
        colors[i] = Color{0, 0, 0};  // black default color for nodes
    }
    write_graph(graph, coords, colors);
}

void IpeOutputWriter::write_graph(const Graph& graph, const Coordinates& coords,
                                  const std::vector<double>& color_scale) {
    std::vector<Color> colors(graph.getNumVertices());
    // scale weights to be between 0 and one
    double maxNodeWeight = 0;
    std::vector<double> scaledWeights(color_scale.size());
    for (NodeId v = 0; v < color_scale.size(); v++) {
        maxNodeWeight = std::max(maxNodeWeight, color_scale[v]);
    }
    for (NodeId v = 0; v < color_scale.size(); v++) {
        scaledWeights[v] = color_scale[v] / maxNodeWeight;
    }
    for (int i = 0; i < graph.getNumVertices(); i++) {
        colors[i] = Common::HSVtoRGB(scaledWeights[i] * 360, 1, 1);  // scale h to be between 0 and 60
    }
    write_graph(graph, coords, colors);
}

void IpeOutputWriter::write_graph(const Graph& graph, const Coordinates& coords, const std::vector<Color>& colors) {
    int n = graph.getNumVertices();

    for (int v = 0; v < n; ++v) {
        std::vector<int> neighbors = graph.getNeighbors(v);
        for (int u : neighbors) {
            if (u < v) continue;
            line(coords[v][0], coords[v][1], coords[u][0], coords[u][1]);
        }
    }
    for (int v = 0; v < n; ++v) {
        point(coords[v][0], coords[v][1], colors[v]);
    }
}

void IpeOutputWriter::new_page() {
    page_end();
    page_start();
}

void IpeOutputWriter::label(const std::string& label, double x, double y, const std::string& color) {
    normalize_point(x, y);
    m_file << R"(<text layer="alpha" transformations="translations" pos=")" << x << " " << y << "\" stroke=\"" << color
           << R"(" type="label" )"
           << R"(halign="center" size="normal" valign="center">)" << label << "</text>\n";
}

void IpeOutputWriter::line(double x1, double y1, double x2, double y2, const std::string& color,
                           const std::string& additional_settings) {
    normalize_point(x1, y1);
    normalize_point(x2, y2);
    m_file << "<path stroke = \"" << color << "\" " << additional_settings << ">\n";
    m_file << x1 << " " << y1 << " m\n";
    m_file << x2 << " " << y2 << " l\n";
    m_file << "</path>\n";
}

void IpeOutputWriter::box(double x1, double y1, double x2, double y2, const std::string& color) {
    normalize_point(x1, y1);
    normalize_point(x2, y2);
    m_file << "<path stroke = \"" << color << "\">\n";
    m_file << x1 << " " << y1 << " m\n"
           << x1 << " " << y2 << " l\n"
           << x2 << " " << y2 << " l\n"
           << x2 << " " << y1 << " l\n"
           << "h\n";
    m_file << "</path>\n";
}

void IpeOutputWriter::point(double x, double y, const std::string& color) {
    normalize_point(x, y);
    m_file << "<use name=\"mark/disk(sx)\" pos=\"" << x << " " << y << R"(" size="small" stroke=")" << color
           << "\"/>\n";
}

void IpeOutputWriter::point(double x, double y, Color color) {
    normalize_point(x, y);
    m_file << "<use name=\"mark/disk(sx)\" pos=\"" << x << " " << y << R"(" size="small" stroke=")" << color.r / 255.0
           << " " << color.g / 255.0 << " " << color.b / 255.0 << "\"/>\n";
}

void IpeOutputWriter::disk(double x, double y, double radius, const std::string& color) {
    normalize_point(x, y);
    m_file << "<path fill=\"" << color << "\" opacity=\"transparent\">\n"
           << radius << " 0 0 " << radius << " " << x << " " << y << " e\n"
           << "</path>\n";
}

void IpeOutputWriter::start_group() { m_file << "<group>\n"; }

void IpeOutputWriter::start_group_with_clipping(double x1, double y1, double x2, double y2) {
    normalize_point(x1, y1);
    normalize_point(x2, y2);
    m_file << "<group clip=\"" << x1 << " " << y1 << " m\n"
           << x1 << " " << y2 << " l\n"
           << x2 << " " << y2 << " l\n"
           << x2 << " " << y1 << " l\n"
           << "h\n"
           << "\">\n";
}

void IpeOutputWriter::end_group() { m_file << "</group>\n"; }

void IpeOutputWriter::file_start() {
    m_file << "<?xml version=\"1.0\"?>\n"
           << "<!DOCTYPE ipe SYSTEM \"ipe.dtd\">\n"
           << "<ipe version=\"70206\" creator=\"Ipe 7.2.6\">\n";
    m_file << "<ipestyle name=\"dummy\">\n";
    m_file << "<symbol name=\"mark/disk(sx)\" transformations=\"translations\">\n"
           << "<path fill=\"sym-stroke\">\n"
           << "0.6 0 0 0.6 0 0 e\n"
           << "</path>\n"
           << "</symbol>\n";
    m_file << "<opacity name=\"60%\" value=\"0.6\"/>\n";
    m_file << "<textstretch name=\"normal\" value=\"0.2\"/>\n";
    m_file << "</ipestyle>\n";
}

void IpeOutputWriter::file_end() { m_file << "</ipe>\n"; }

void IpeOutputWriter::page_start() {
    m_file << "<page>\n";
    m_file << "<layer name=\"alpha\"/>\n";
    m_file << "<view layers=\"alpha\" active=\"alpha\"/>\n";
}

void IpeOutputWriter::page_end() { m_file << "</page>\n"; }

void IpeOutputWriter::normalize_point(double& x, double& y) {
    x = x * m_scaling + m_margin;
    y = y * m_scaling + m_margin;
}
