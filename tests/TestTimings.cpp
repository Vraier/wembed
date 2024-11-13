#include <iostream>

#include <gtest/gtest.h>

#include "Timings.hpp"
#include "Macros.hpp"

using Timer = util::Timer;
using TimingResult = util::TimingResult;

TEST(Timings, hierarchicalConstructionTest) {
    Timer timer;
    timer.startTiming("a", "A");
    timer.startTiming("a1", "A1");
    timer.stopTiming("a1");
    timer.startTiming("a2", "A2");
    timer.startTiming("a2I", "A2I");
    timer.stopTiming("a2I");
    timer.stopTiming("a2");
    timer.stopTiming("a");
    timer.startTiming("b", "B");
    timer.stopTiming("b");
    timer.startTiming("c", "C");
    timer.startTiming("c1", "C1");
    timer.stopTiming("c1");
    timer.stopTiming("c");

    auto result = timer.getHierarchicalTimingResults();
    EXPECT_EQ(result[0].depth, 0);
    EXPECT_EQ(result[0].display_name, "A");
    EXPECT_EQ(result[1].depth, 1);
    EXPECT_EQ(result[1].display_name, "A1");
    EXPECT_EQ(result[2].depth, 1);
    EXPECT_EQ(result[2].display_name, "A2");
    EXPECT_EQ(result[3].depth, 2);
    EXPECT_EQ(result[3].display_name, "A2I");
    EXPECT_EQ(result[4].depth, 0);
    EXPECT_EQ(result[4].display_name, "B");
    EXPECT_EQ(result[5].depth, 0);
    EXPECT_EQ(result[5].display_name, "C");
    EXPECT_EQ(result[6].depth, 1);
    EXPECT_EQ(result[6].display_name, "C1");

    std::cout << util::timingsToStringRepresentation(result) << std::endl;
}
