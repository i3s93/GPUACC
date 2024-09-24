target_sources(Boltzmann PRIVATE
                statistics.hpp
                parameters.hpp
                sincc.hpp
                Solver_Manager.hpp
                gauss_legendre.hpp
                gauss_legendre.cpp)

find_package(GSL REQUIRED)

target_link_libraries(Boltzmann PUBLIC GSL::gsl GSL::gslcblas)
