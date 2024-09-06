#ifndef GAUSS_LEGENDRE_HPP
#define GAUSS_LEGENDRE_HPP

#include <vector>
#include <gsl/gsl_integration.h>

// Function to retrieve the Gauss-Legendre nodes and weights on the interval [a,b]
void get_gauss_legendre_rule(int n, std::vector<double>& wts, std::vector<double>& nodes, double a, double b);

#endif // GAUSS_LEGENDRE_HPP
