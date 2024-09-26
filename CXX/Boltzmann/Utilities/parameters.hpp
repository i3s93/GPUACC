#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

// Maps from a tuple to linear index assuming a row-major ordering
#define IDX2(i1, i2, N1, N2) (i1*N2 + i2)
#define IDX3(i1, i2, i3, N1, N2, N3) ((i1*N2+i2)*N3 + i3)
#define IDX4(i1, i2, i3, i4, N1, N2, N3, N4) (((i1*N2 + i2)*N3 + i3)*N4 + i4)
#define IDX5(i1, i2, i3, i4, i5, N1, N2, N3, N4, N5) ((((i1*N2 + i2)*N3 + i3)*N4 + i4)*N5 + i5)

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
