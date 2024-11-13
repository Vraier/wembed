#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

namespace util {

struct TimingResult {
    size_t depth;
    std::string display_name;
    double value;
};

/**
 * Converts timing results to a pretty-printed string.
*/
std::string timingsToStringRepresentation(const std::vector<TimingResult>& timings);

/**
 * Used to represent timings as a tree. Each node is the summed time for a specific
 * task. A parent is the sum of the child timings plus any remaining time.
*/
class Timer {
   private:
    using Timepoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

    struct TimingEntry {
        std::string parent;
        std::string key;
        std::string display_name;
        double value;
    };

   public:
    /**
        * Starts a new timing. If there is already a running timing, the new timing is
        * created as a child.
    */
    void startTiming(const std::string& key, const std::string& display_name);

    void stopTiming(const std::string& key);

    void reset();

    std::vector<TimingResult> getHierarchicalTimingResults() const;

   private:
    void insertCurrentTiming(const std::string& parent, const std::string& key, const std::string& display_name);

    void getHierarchicalTimingResultsImpl(size_t depth, const std::string& key, std::vector<TimingResult>& results) const;

    std::vector<std::pair<std::string, Timepoint>> timing_stack;
    std::vector<TimingEntry> entries;
    std::unordered_map<std::string, size_t> key_to_entry;
};

}  // namespace util
