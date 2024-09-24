#ifndef SINCC_HPP
#define SINCC_HPP

#include <cmath>
#include <limits>

template<typename T>
T sincc(T x){
    T eps = std::numeric_limits<T>::epsilon();
    return std::sin(x + eps)/(x + eps);
}

#endif // SINCC_HPP
