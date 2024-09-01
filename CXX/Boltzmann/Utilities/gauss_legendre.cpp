#include "gauss_legendre.hpp"

// Evaluates the Legendre polynomial of degree n at the location x using recusion
double legendre(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return x;

    double P0 = 1.0;
    double P1 = x;
    double P2;

    for (int k = 2; k <= n; ++k) {
        P2 = ((2.0 * k - 1.0) * x * P1 - (k - 1.0) * P0) / k;
        P0 = P1;
        P1 = P2;
    }

    return P2;
}

// Function to compute the Gauss-Legendre nodes and weights
void gauss_legendre(int n, std::vector<double>& nodes, std::vector<double>& weights) {

    nodes.resize(n);
    weights.resize(n);

    // Initialize variables
    double tolerance = 1e-14;
    int max_iter = 100;

    // Loop over the roots (nodes)
    for (int i = 0; i < n; ++i) {
        // Initial guess for the ith root (node)
        double x = std::cos(M_PI * (i + 0.75) / (n + 0.5));

        // Use Newton's method to find the root
        double error;
        int iter = 0;

        while (std::abs(error) > tolerance && iter < max_iter) {
            double Pn = legendre(n, x);
            double Pn_prime = (x * Pn - legendre(n - 1, x)) * n / (x * x - 1.0);
            error = Pn / Pn_prime;
            x -= error;
            iter++;
        }

        // Store the root (node) and corresponding weight
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x * x) * std::pow(legendre(n - 1, x), 2));
    }
}