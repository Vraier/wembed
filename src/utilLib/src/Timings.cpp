#include "Timings.hpp"

#include <sstream>
#include <iomanip>

#include "Macros.hpp"

namespace util {
    void Timer::startTiming(const std::string& key, const std::string& display_name) {
        ASSERT(key != "", "invalid key");
        std::string parent("");
        if (!timing_stack.empty()) {
            parent = timing_stack.back().first;
        }
        insertCurrentTiming(parent, key, display_name);
        timing_stack.emplace_back(key, std::chrono::high_resolution_clock::now());
    }

    void Timer::stopTiming(const std::string& key) {
        ASSERT(!timing_stack.empty());
        const auto& [old_key, start] = timing_stack.back();
        ASSERT(old_key == key, "stopped the wrong timer: running is " << old_key << ", but provided was " << key);

        auto map_it = key_to_entry.find(key);
        ASSERT(map_it != key_to_entry.end());
        Timepoint end = std::chrono::high_resolution_clock::now();
        entries[map_it->second].value += std::chrono::duration<double>(end - start).count();
        timing_stack.pop_back();
    }

    void Timer::reset() {
        timing_stack.clear();
        entries.clear();
        key_to_entry.clear();
    }

    void Timer::insertCurrentTiming(const std::string& parent, const std::string& key, const std::string& display_name) {
        auto map_it = key_to_entry.find(key);
        if (map_it != key_to_entry.end()) {
            const TimingEntry& entry = entries[map_it->second];
            ASSERT(parent == entry.parent && key == entry.key && display_name == entry.display_name,
                   "A different timing with this key already exists: " << key);
        } else {
            key_to_entry[key] = entries.size();
            entries.emplace_back(parent, key, display_name, 0);
        }
    }

    std::vector<TimingResult> Timer::getHierarchicalTimingResults() const {
        std::vector<TimingResult> results;
        getHierarchicalTimingResultsImpl(0, "", results);
        return results;
    }

    void Timer::getHierarchicalTimingResultsImpl(size_t depth, const std::string& key, std::vector<TimingResult>& results) const {
        for (const auto& [parent_key, child_key, name, value]: entries) {
            if (parent_key == key) {
                results.emplace_back(depth, name, value);
                getHierarchicalTimingResultsImpl(depth + 1, child_key, results);
            }
        }
    }


    std::string timingsToStringRepresentation(const std::vector<TimingResult>& timings) {
        std::ostringstream builder;
        for (const TimingResult& timing: timings) {
            for (size_t i = 0; i < timing.depth; i++) {
                builder << "   ";
            }
            std::ostringstream number;
            number << std::setprecision(4) << timing.value << "s";
            builder << "+- " << std::left << std::setw(15) << number.str() << timing.display_name;
            builder << std::endl;
        }
        return builder.str();
    }
}  // namespace util
