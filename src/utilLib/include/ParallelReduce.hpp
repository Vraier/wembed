#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace util {

/**
 * Deterministic parallel reductions.
 *
 * A plain OpenMP `reduction(+:x)` combines the per-thread partial sums in an
 * unspecified order, This breaks determinism on float numbers.
 * 
 * This function implements a block-reduce pattern that is deterministic. Regardless of thread count.
 */

// Sums element(i) for i in [0, n). element must be safe to call concurrently.
// For a per-dimension sum, call this once per dimension: each call fixes the same
// block boundaries and combine order, so every column is reduced deterministically.
template <typename ElementFn>
inline double deterministicSum(std::size_t n, ElementFn&& element, std::size_t blockSize = 4096) {
    if (n == 0) return 0.0;
    const std::size_t numBlocks = (n + blockSize - 1) / blockSize;
    std::vector<double> partial(numBlocks, 0.0);
#pragma omp parallel for schedule(static)
    for (std::size_t b = 0; b < numBlocks; b++) {
        const std::size_t begin = b * blockSize;
        const std::size_t end = std::min(begin + blockSize, n);
        double sum = 0.0;
        for (std::size_t i = begin; i < end; i++) {
            sum += element(i);
        }
        partial[b] = sum;
    }
    double total = 0.0;
    for (std::size_t b = 0; b < numBlocks; b++) {
        total += partial[b];
    }
    return total;
}

}  // namespace util
