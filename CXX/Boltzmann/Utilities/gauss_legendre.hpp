#ifndef GAUSS_LEGENDRE_HPP
#define GAUSS_LEGENDRE_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>  
#include <numeric> 

// Evaluates the Legendre polynomial of degree n at the location x using recusion
double legendre(int n, double x);

// Function to compute the Gauss-Legendre nodes and weights
void gauss_legendre(int n, std::vector<double>& nodes, std::vector<double>& weights);

#endif // GAUSS_LEGENDRE_HPP