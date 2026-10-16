#pragma once

#include <gtest/gtest.h>
#include <type_traits>

#include "wembed_types.hpp"

using wembed::flt_t;

static void inline EXPECT_FLT_T_EQ(flt_t A, flt_t B) {
    if constexpr (std::is_same_v<flt_t, float>) {
        EXPECT_FLOAT_EQ(A, B);
    } else {
        EXPECT_DOUBLE_EQ(A, B);
    }
}

static void inline ASSERT_FLT_T_EQ(flt_t A, flt_t B) {
    if constexpr (std::is_same_v<flt_t, float>) {
        ASSERT_FLOAT_EQ(A, B);
    } else {
        ASSERT_DOUBLE_EQ(A, B);
    }
}
