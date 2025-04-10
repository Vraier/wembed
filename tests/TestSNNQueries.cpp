#include <gtest/gtest.h>

#include "SNNQueries.hpp"
#include "VecList.hpp"

TEST(SNNQueries, basic) {
    using NodeId = int;

    VecList list(3);
    list.setSize(5);
    VecRef first = list[0];
    first[0] = 0.5;
    first[1] = 2.0;
    first[2] = 3.0;
    std::vector<std::pair<CVecRef, NodeId>> data{ {list[0], 0} };
    SNNQueries snn(data, 3);
    std::vector<NodeId> buffer;
    snn.query_sphere(list[0], 0.1, buffer);
    EXPECT_EQ(buffer.size(), 1);
    EXPECT_EQ(buffer[0], 0);

    VecRef second = list[1];
    second[0] = 0.0;
    second[1] = 1.0;
    second[2] = 3.0;
    VecRef third = list[2];
    third[0] = 0.0;
    third[1] = 1.0;
    third[2] = 0.0;
    data.emplace_back(list[1], 2);
    data.emplace_back(list[2], 1);
    snn = SNNQueries(data, 3);

    {
        buffer.clear();
        VecRef query = list[3];
        query[0] = 0.0;
        query[1] = 1.5;
        query[2] = 3.1;
        snn.query_sphere(std::move(query), 1.0, buffer);
        EXPECT_EQ(buffer.size(), 2);
        std::sort(buffer.begin(), buffer.end());
        EXPECT_EQ(buffer[0], 0);
        EXPECT_EQ(buffer[1], 2);
    }

    {
        buffer.clear();
        VecRef query = list[3];
        snn.query_sphere(std::move(query), 0.5, buffer);
        EXPECT_EQ(buffer.size(), 0);
    }

    {
        buffer.clear();
        VecRef query = list[3];
        query[0] = 0.5;
        query[1] = 0.0;
        query[2] = 4.0;
        snn.query_sphere(std::move(query), 1.7, buffer);
        EXPECT_EQ(buffer.size(), 1);
        EXPECT_EQ(buffer[0], 2);
    }
}
