#include <gtest/gtest.h>

#include <memory>

#include "EdgeDetection.hpp"
#include "Euclidean.hpp"
#include "GeneralGraphInfo.hpp"
#include "Reconstruction.hpp"

#define EPSILON 0.0001

TEST(Metric, GeneralGrapInfo) {
    std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    Graph g(edges);

    GeneralGraphInfo metric(g);

    std::vector<std::string> names = metric.getMetricNames();
    std::vector<std::string> values = metric.getMetricValues();

    EXPECT_EQ(names.size(), values.size());
    EXPECT_EQ(names[0], "num_nodes");
    EXPECT_EQ(std::stoi(values[0]), 5);

    EXPECT_EQ(names[1], "num_edges");
    EXPECT_EQ(std::stoi(values[1]), 4);
}

TEST(Metric, Reconstruction) {
    std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}};
    Graph g(edges);

    // construct embedding
    std::vector<std::vector<double>> coords = {{0, 0}, {1, 0}, {2, 0}};
    double nodeSampleFraction = 1.0;
    std::shared_ptr<Embedding> embedding = std::make_unique<Euclidean>(coords);

    Reconstruction metric(g, embedding, nodeSampleFraction);

    std::vector<std::string> names = metric.getMetricNames();
    std::vector<std::string> values = metric.getMetricValues();

    EXPECT_EQ(names.size(), values.size());
    EXPECT_EQ(names[0], "constructDeg");
    EXPECT_EQ(names[1], "MAP");

    EXPECT_NEAR(std::stod(values[0]), 1.0, EPSILON);
    EXPECT_NEAR(std::stod(values[1]), 1.0, EPSILON);

    // construct bad embedding
    std::vector<std::vector<double>> badCoords = {{0, 0}, {3, 0}, {1, 0}};
    std::shared_ptr<Embedding> badEmbedding = std::make_unique<Euclidean>(badCoords);

    Reconstruction badMetric(g, badEmbedding, nodeSampleFraction);

    std::vector<std::string> badValues = badMetric.getMetricValues();
    EXPECT_NEAR(std::stod(badValues[0]), 1.0/3.0, EPSILON);
    EXPECT_NEAR(std::stod(badValues[1]), (0.5+0.5+1)/3.0, EPSILON);
}

TEST(Metric, EdgeDetection) {
    std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}};
    Graph g(edges);

    // construct embedding
    std::vector<std::vector<double>> coords = {{0, 0}, {1, 0}, {2, 0}};
    double edgeSampleScale = 1.0;
    std::shared_ptr<Embedding> embedding = std::make_unique<Euclidean>(coords);

    EdgeDetection metric(g, embedding, edgeSampleScale);

    std::vector<std::string> names = metric.getMetricNames();
    std::vector<std::string> values = metric.getMetricValues();

    EXPECT_EQ(names.size(), values.size());
    EXPECT_EQ(names[0], "precision");
    EXPECT_EQ(names[1], "recall");
    EXPECT_EQ(names[2], "edgeF1");
    EXPECT_NEAR(std::stod(values[0]), 1.0, EPSILON);
    EXPECT_NEAR(std::stod(values[1]), 1.0, EPSILON);
    EXPECT_NEAR(std::stod(values[2]), 1.0, EPSILON);

    // construct bad embedding
    std::vector<std::vector<double>> badCoords = {{0, 0}, {3, 0}, {1, 0}};
    std::shared_ptr<Embedding> badEmbedding = std::make_unique<Euclidean>(badCoords);

    EdgeDetection badMetric(g, badEmbedding, edgeSampleScale);

    std::vector<std::string> badValues = badMetric.getMetricValues();
    EXPECT_NEAR(std::stod(badValues[0]), 2.0/3.0, EPSILON);
    EXPECT_NEAR(std::stod(badValues[1]), 1.0, EPSILON);
    EXPECT_NEAR(std::stod(badValues[2]), 2.0/(1.0/std::stod(badValues[0]) + 1.0/std::stod(badValues[1])), EPSILON);
}