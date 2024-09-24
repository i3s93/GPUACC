#ifndef BOLTZMANN_COLLISIONS_OMP_HPP
#define BOLTZMANN_COLLISIONS_OMP_HPP

#include <cmath>
#include <complex>
#include <vector>
#include <iostream>
#include <omp.h>
#include <fftw3.h>

#include "../Utilities/Solver_Manager.hpp"
#include "../Utilities/parameters.hpp"
#include "../Utilities/sincc.hpp"

void boltzmann_vhs_spectral_solver_omp(std::vector<double> &Q, 
                                   std::vector<double> &f_in, 
                                   const SolverManager &sm, const SolverParameters &sp, 
                                   const double b_gamma, const double gamma);

#endif // BOLTZMANN_COLLISIONS_OMP_HPP
