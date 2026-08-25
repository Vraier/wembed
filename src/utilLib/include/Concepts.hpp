#pragma once
#include <type_traits>

template <typename T>
concept FLT_T = std::is_same_v<T, float> || std::is_same_v<T, double>;
