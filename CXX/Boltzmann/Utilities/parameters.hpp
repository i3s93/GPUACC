#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

// Map from tuple to index assuming Nv points are used in each dimension using row-major ordering
#define IDX(i1, i2, i3, N1, N2, N3) ((i1*N2+i2)*N3 + i3)

constexpr double pi = 3.14159265358979323846;

struct SolverParameters
{
    // Spectral solver cutoff parameters
    double S;
    double R;
    double L;

    // Quadature parameters per dim
    int Nv;
    int Nr;
    int Ns;
};


#endif // PARAMETERS_HPP
