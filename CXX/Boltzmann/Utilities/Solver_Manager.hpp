#ifndef SOLVER_MANAGER_HPP
#define SOLVER_MANAGER_HPP

#include <vector>
#include <complex>

struct SolverManager
{
    // 1D wave numbers for vx, vy, vz
    std::vector<int> l1;
    std::vector<int> l2;
    std::vector<int> l3;

    // Gauss-Legendre quadrature weights and nodes
    std::vector<double> wts_gl;
    std::vector<double> nodes_gl;

    // Spherical quadrature weights and nodes
    std::vector<double> wts_sph;
    std::vector<double> sigma1_sph;
    std::vector<double> sigma2_sph;
    std::vector<double> sigma3_sph;

    // Precomputed weights for transform data
    std::complex<double>* alpha1;
    double* beta1;
    double* beta2;

};


#endif // SOLVER_MANAGER_HPP
