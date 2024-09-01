#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

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