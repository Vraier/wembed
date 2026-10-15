#pragma once

#include <vector>

#include "VecList.hpp"

class Optimizer {
   public:
    virtual ~Optimizer() {}

    virtual void update(VecList<>& parameters, const VecList<>& gradients, flt_t learningRate) = 0;
    virtual void reset() = 0;
};
