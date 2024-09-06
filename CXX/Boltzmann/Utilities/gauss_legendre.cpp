#include "gauss_legendre.hpp"

void get_gauss_legendre_rule(int n, std::vector<double>& wts, std::vector<double>& nodes, double a, double b) {

    // Weights will be extracted from the table into vectors
    wts.resize(n);
    nodes.resize(n);

     // Allocate memory for the GSL table
    gsl_integration_glfixed_table* table = gsl_integration_glfixed_table_alloc(n);

    // Extract the nodes and weights from the GSL table and store them in vectors
    for (int i = 0; i < n; ++i) {
        gsl_integration_glfixed_point(a, b, i, &nodes[i], &wts[i], table);
    }

    // Free the GSL table memory
    gsl_integration_glfixed_table_free(table);

    return;
}
