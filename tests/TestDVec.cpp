#include <gtest/gtest.h>

#include "DVec.hpp"
#include "VecList.hpp"

// test initialization
TEST(VecTests, Initialization) {
    VecRef::MemoryType mem1[VecRef::numEntriesForDimension(2)];
    VecRef(&mem1[0], 2).setAll(0);
    CVecRef u(&mem1[0], 2);
    ASSERT_DOUBLE_EQ(u[0], 0);

    VecRef::MemoryType mem2[VecRef::numEntriesForDimension(2)];
    VecRef v(&mem2[0], 2);
    v[0] = 1.0;
    v[1] = 2.0;

    ASSERT_DOUBLE_EQ(v[0], 1.0);
    ASSERT_DOUBLE_EQ(v[1], 2.0);
    ASSERT_EQ(v.dimension(), 2);
}

// test arithmetic operations
TEST(VecTests, Arithmetic) {
    VecRef::MemoryType mem1[VecRef::numEntriesForDimension(2)];
    VecRef(&mem1[0], 2).setAll(0);
    CVecRef u(&mem1[0], 2);
    ASSERT_EQ(u[0], 0);

    VecRef::MemoryType mem2[VecRef::numEntriesForDimension(2)];
    VecRef v(&mem2[0], 2);
    v[0] = 1.0;
    v[1] = 2.0;

    v = u + v;
    auto x = 3 * (u + v);
    v = x - v;
    ASSERT_DOUBLE_EQ(v[0], 2);
    ASSERT_DOUBLE_EQ(v[1], 4);
    ASSERT_EQ(v.dimension(), 2);
}

// test move assignment and copy construction
TEST(VecTests, MoveAndCopy) {
    VecRef::MemoryType mem1[VecRef::numEntriesForDimension(2)];
    VecRef(&mem1[0], 2).setAll(0);
    CVecRef u(&mem1[0], 2);
    ASSERT_DOUBLE_EQ(u[0], 0);

    VecRef::MemoryType mem2[VecRef::numEntriesForDimension(2)];
    VecRef v(&mem2[0], 2);
    v[0] = 1.0;
    v[1] = 2.0;

    u = std::move(v);
    CVecRef u2 = u;
    ASSERT_DOUBLE_EQ(u[0], 1.0);
    ASSERT_DOUBLE_EQ(u2[0], 1.0);
    ASSERT_DOUBLE_EQ(u[1], 2.0);
    ASSERT_DOUBLE_EQ(u2[1], 2.0);
}

// test VecList
TEST(VecTests, VecList) {
    VecBuffer<3> buffer(2);
    TmpVec<0> a(buffer, 1.0);
    {
        TmpVec<1> b(buffer);
        b[0] = -1.0;
        b[1] = 1.5;
        a += 3 * b;
        ASSERT_DOUBLE_EQ(a[0], -2);
        ASSERT_DOUBLE_EQ(a[1], 5.5);
        TmpCVec<1> tmp1 = std::move(b);
        TmpCVec<1> tmp2 = tmp1;
    }

    TmpVec<1> c(buffer, 1.0);
    a -= c;
    ASSERT_DOUBLE_EQ(a[0], -3);
    ASSERT_DOUBLE_EQ(a[1], 4.5);

    TmpVec<2> random(buffer);
    random.setToRandomUnitVector();

    VecList list(2);
    list.setSize(4);
    list[0] = a;
    list[1] = c;
    list[2] = random;

    CVecRef first = list[0];
    list[2] += 2 * first;
    VecRef last = list[3];
    last = first;

    ASSERT_DOUBLE_EQ(list[0][0], -3);
    ASSERT_DOUBLE_EQ(list[0][1], 4.5);
    ASSERT_DOUBLE_EQ(list[1][0], 1);
    ASSERT_DOUBLE_EQ(list[1][1], 1);
}
