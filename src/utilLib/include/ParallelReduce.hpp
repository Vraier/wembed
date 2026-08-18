#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace util {

/**
 * Deterministic parallel reductions.
 *
 * A plain OpenMP `reduction(+:x)` combines the per-thread partial sums in an
 * unspecified order, which breaks determinism for floating-point sums. This uses
 * a fixed block-reduce pattern that gives the same result regardless of thread
 * count. For a per-dimension sum, call this once per dimension.
 */
template <typename ElementFn>
inline float deterministicSum(std::size_t n, ElementFn&& element, std::size_t blockSize = 4096) {
    if (n == 0) return 0.0;
    const std::size_t numBlocks = (n + blockSize - 1) / blockSize;
    std::vector<float> partial(numBlocks, 0.0);
#pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < numBlocks; b++) {
        const std::size_t begin = b * blockSize;
        const std::size_t end = std::min(begin + blockSize, n);
        float sum = 0.0;
        for (std::size_t i = begin; i < end; i++) {
            sum += element(i);
        }
        partial[b] = sum;
    }
    float total = 0.0;
    for (std::size_t b = 0; b < numBlocks; b++) {
        total += partial[b];
    }
    return total;
}

}  // namespace util
