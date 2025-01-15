#pragma once

#include <vector>

#include "DVec.hpp"
#include "Macros.hpp"

class VecList {
    using Memory = VecRef::MemoryType;
    // TODO: support iteration

   public:
    VecList(unsigned int dimension) : num_elements(0), dim(dimension), data() {}
    VecList(unsigned int dimension, size_t size) : num_elements(size), dim(dimension), data() {
        setSize(num_elements, 0.0);
    }

    VecList(std::vector<std::vector<double>> list) : num_elements(list.size()), dim(list[0].size()), data() {
        setSize(num_elements);
        for (size_t i = 0; i < list.size(); ++i) {
            ASSERT(list[i].size() == dim, "list[" << i << "].size()=" << list[i].size() << ", dim=" << dim);
            for (unsigned int d = 0; d < dim; ++d) {
                (*this)[i][d] = list[i][d];
            }
        }
    }

    void setSize(size_t new_size) {
        num_elements = new_size;
        data.resize(num_elements * VecRef::numEntriesForDimension(dim));
    }

    void setSize(size_t new_size, double default_value) {
        // ASSERT(size() == 0);
        setSize(new_size);
        for (size_t i = 0; i < size(); ++i) {
            (*this)[i].setAll(default_value);
        }
    }

    void setAll(double default_value) {
        #pragma omp parallel for
        for (size_t i = 0; i < size(); ++i) {
            (*this)[i].setAll(default_value);
        }
    }

    size_t size() const { return num_elements; }

    unsigned int dimension() const { return dim; }

    ALWAYS_INLINE VecRef operator[](size_t i) {
        ASSERT(i < size(), "i=" << i << ", size=" << size());
        size_t index = i * VecRef::numEntriesForDimension(dim);
        return VecRef(&data[index], dim);
    }

    ALWAYS_INLINE CVecRef operator[](size_t i) const {
        ASSERT(i < size(), "i=" << i << ", size=" << size());
        size_t index = i * VecRef::numEntriesForDimension(dim);
        return CVecRef(&data[index], dim);
    }

    std::vector<std::vector<double>> convertToVector() const {
        std::vector<std::vector<double>> result(size(), std::vector<double>(dimension()));
        for (size_t i = 0; i < size(); ++i) {
            for (unsigned int d = 0; d < dimension(); d++) {
                result[i][d] = (*this)[i][d];
            }
        }
        return result;
    }

   private:
    size_t num_elements;
    unsigned int dim;
    std::vector<Memory> data;
};
