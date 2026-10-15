#include <gtest/gtest.h>

#include <stdexcept>

#include "wembed.h"

TEST(PublicInterface, NeighborhoodsKeepIsolatedVertices) {
    wembed::Graph g = wembed::graphFromNeighborhoods({0, 1, 3, 3, 3, 4}, {1, 0, 2, 1});
    EXPECT_EQ(g.getNumVertices(), 5);
    EXPECT_EQ(g.getNumEdges(), 3);
    EXPECT_EQ(g.getNumNeighbors(2), 1);
    EXPECT_EQ(g.getNumNeighbors(3), 0);
    EXPECT_TRUE(g.areNeighbors(4, 1));
}

TEST(PublicInterface, NeighborhoodsIgnoreSelfLoopsAndDuplicates) {
    wembed::Graph g = wembed::graphFromNeighborhoods({0, 3, 4}, {0, 1, 1, 0});
    EXPECT_EQ(g.getNumVertices(), 2);
    EXPECT_EQ(g.getNumEdges(), 1);
}

TEST(PublicInterface, NeighborhoodsMatchEdges) {
    wembed::Graph fromEdges = wembed::graphFromEdges({{0, 1}, {1, 2}, {3, 0}}, 5);
    wembed::Graph fromNeighborhoods = wembed::graphFromNeighborhoods({0, 2, 3, 3, 3, 3}, {1, 3, 2});
    EXPECT_EQ(fromEdges.getNumVertices(), fromNeighborhoods.getNumVertices());
    EXPECT_EQ(fromEdges.toString(), fromNeighborhoods.toString());
}

TEST(PublicInterface, NeighborhoodsOutOfRangeThrows) {
    EXPECT_THROW(wembed::graphFromNeighborhoods({0, 1, 2}, {1, 2}), std::invalid_argument);
    EXPECT_THROW(wembed::graphFromNeighborhoods({0, 1}, {-1}), std::invalid_argument);
}

TEST(PublicInterface, NeighborhoodsInvalidOffsetsThrow) {
    EXPECT_THROW(wembed::graphFromNeighborhoods({}, {}), std::invalid_argument);
    EXPECT_THROW(wembed::graphFromNeighborhoods({1, 1}, {0}), std::invalid_argument);
    EXPECT_THROW(wembed::graphFromNeighborhoods({0, 2, 1}, {1}), std::invalid_argument);
    EXPECT_THROW(wembed::graphFromNeighborhoods({0, 1}, {0, 0}), std::invalid_argument);
    EXPECT_THROW(wembed::graphFromNeighborhoods({0, -1}, {}), std::invalid_argument);
}

TEST(PublicInterface, EmptyNeighborhoods) {
    wembed::Graph g = wembed::graphFromNeighborhoods({0}, {});
    EXPECT_EQ(g.getNumVertices(), 0);
}
