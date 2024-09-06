#ifndef BOLTZMANN_COLLISIONS_HPP
#define BOLTZMANN_COLLISIONS_HPP

#include <cmath>
#include <complex>
#include <vector>
#include <limits>
#include <iostream>
#include <fftw3.h>

#include "../Utilities/Solver_Manager.hpp"
#include "../Utilities/parameters.hpp"

template<typename T>
T sincc(T x){
    T eps = std::numeric_limits<T>::epsilon();
    return std::sin(x + eps)/(x + eps);
}

void boltzmann_vhs_spectral_solver(std::vector<double> &Q, 
                                   std::vector<double> &f_in, 
                                   const SolverManager &sm, const SolverParameters &sp, 
                                   const double b_gamma, const double gamma);

#endif // BOLTZMANN_COLLISIONS_HPP
