#pragma once

#include <iostream>
#include <string>
#include <cstring>

#ifndef NDEBUG
#ifndef EMBEDDING_USE_ASSERTIONS
#define EMBEDDING_USE_ASSERTIONS
#endif
#endif

using ll = long long;

// suppress warnings for unused variables (for example when using in assertions)
template <typename T>
void unused(T&&) {}

// logging macro
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#ifdef NDEBUG
#define LOG_DEBUG(msg)
#define FILE_INFO ""
#else
#define FILE_INFO __FILENAME__ << "(" << __LINE__ << ", " << __FUNCTION__ << "): "
#define LOG_DEBUG(msg) std::cout << "[DEBUG] " << FILE_INFO << msg << std::endl;
#endif
#define LOG_INFO(msg) std::cout << "[INFO] " << FILE_INFO << msg << std::endl;
#define LOG_WARNING(msg) std::cout << "[WARNING] " << FILE_INFO << msg << std::endl;
#define LOG_ERROR(msg) std::cout << "[ERROR] " << FILE_INFO << msg << std::endl; std::abort();

// assertion with variable number of arguments
#ifdef EMBEDDING_USE_ASSERTIONS
#define ASSERT_2(cond, msg)                                     \
    do {                                                        \
        if (!(cond)) {                                          \
            LOG_ERROR("Assertion `" #cond "` failed: " << msg); \
            std::abort();                                       \
        }                                                       \
    } while (0)

#define ASSERT_1(cond) ASSERT_2(cond, "")
#else
#define ASSERT_2(cond, msg) unused(cond)
#define ASSERT_1(cond) unused(cond)
#endif

#define NARG(...) NARG_(__VA_ARGS__, RSEQ_N())
#define NARG_(...) ARG_N(__VA_ARGS__)
#define ARG_N(_1, _2, N, ...) N
#define RSEQ_N() 2, 1, 0

#define ASSERT_(N) ASSERT_##N
#define ASSERT_EVAL(N) ASSERT_(N)
#define ASSERT(...) ASSERT_EVAL(NARG(__VA_ARGS__))(__VA_ARGS__)
#define ASSERT_CLOSE(x, y) ASSERT(((x) - (y)) * ((x) - (y)) < 0.001 * 0.001)

#ifdef EMBEDDING_USE_ASSERTIONS
#define ASSERTIONS_ACTIVE true
#else
#define ASSERTIONS_ACTIVE false
#endif

// inlining
#if defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define ALWAYS_INLINE
#endif

// optimization hint
#ifdef EMBEDDING_USE_ASSERTIONS
#define OPTIMIZATION_HINT(cond) ASSERT(cond)
#else
#if defined(__GNUC__) || defined(__clang__)
#define OPTIMIZATION_HINT(cond)      \
    do {                             \
        if (!(cond)) {               \
            __builtin_unreachable(); \
        }                            \
    } while (0)
#else
#define OPTIMIZATION_HINT
#endif
#endif
