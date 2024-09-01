#include "spherical_design.hpp"

SphericalDesign get_spherical_design(int N) {

    std::string filename;
    
    switch (N) {

        case 6:   filename = "ss003.006.txt"; break;
        case 12:  filename = "ss005.012.txt"; break;
        case 32:  filename = "ss007.032.txt"; break;
        case 48:  filename = "ss009.048.txt"; break;
        case 70:  filename = "ss011.070.txt"; break;
        case 94:  filename = "ss013.094.txt"; break;
        case 120: filename = "ss015.120.txt"; break;
        case 156: filename = "ss017.156.txt"; break;
        case 192: filename = "ss019.192.txt"; break;
        default: 
            throw std::invalid_argument("Invalid value of N");

    }

    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error("Could not open file " + filename);
    }

    SphericalDesign sd;
    std::string line;

    std::istringstream iss;
    double x_node, y_node, z_node;

    // Read each line of the file and extract the values on each of the lines
    while (std::getline(file, line)){

        iss.clear();
        iss.str(line);
        iss >> x_node >> y_node >> z_node;

        sd.x.push_back(x_node);
        sd.y.push_back(y_node);
        sd.z.push_back(z_node);

    }

    return sd;
}

void print_spherical_design(const SphericalDesign & sd){

    assert(sd.x.size() == sd.y.size() && sd.y.size() == sd.z.size() &&
           "Vectors x, y, and z must all have the same size");

    // Extract the number of quadrature points
    int Ns = sd.x.size();

    std::cout << "\nQuadrature points being used: " << Ns << "\n";

    for (int i = 0; i < Ns; ++i){
        std::cout << sd.x[i] << sd.y[i] << sd.z[i] << "\n";
    }

}