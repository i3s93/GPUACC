#ifndef SPHERICAL_DESIGN_HPP
#define SPHERICAL_DESIGN_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>
#include <stdexcept>

struct SphericalDesign {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
};

SphericalDesign get_spherical_design(int N);

void print_spherical_design(const SphericalDesign & sd);

#endif // SPHERICAL_DESIGN_HPP