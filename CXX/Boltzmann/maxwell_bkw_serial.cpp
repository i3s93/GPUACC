#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

#include <tclap/CmdLine.h>
#include <omp.h>

#include "SphericalDesign/spherical_design.hpp"
#include "Utilities/parameters.hpp"
#include "Utilities/statistics.hpp"
#include "Utilities/Solver_Manager.hpp"
#include "Utilities/gauss_legendre.hpp"
#include "Collisions/boltzmann_collisions_serial.hpp"

int main(int argc, char** argv) {

    int Nv, Ns, trials;

    try {
        // Create each of the arguments
        TCLAP::CmdLine cmd("Command description message", ' ', "1.0");
        TCLAP::ValueArg<int> Nv_Arg("", "Nv", "Number of points per dimension in velocity", false, 32, "int");
        TCLAP::ValueArg<int> Ns_Arg("", "Ns", "Number of points on the unit sphere", false, 12, "int");
        TCLAP::ValueArg<int> t_Arg("t", "trials", "Number of trials to use for statistics", false, 1, "int");

        cmd.add(Nv_Arg);
        cmd.add(Ns_Arg);
        cmd.add(t_Arg);

        // Parse the argv array.
        cmd.parse(argc, argv);

        // Assign parsed values to variables
        Nv = Nv_Arg.getValue();
        Ns = Ns_Arg.getValue();
        trials = t_Arg.getValue();

        std::cout << "\nRun arguments:" << "\n";
        std::cout << "Nv = " << Nv << "\n";
        std::cout << "Ns = " << Ns << "\n";
        std::cout << "trials = " << trials << "\n";
    } catch (TCLAP::ArgException &e)
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << "\n"; }

    // Test for Maxwell molecules
    const double gamma = 0;
    const double b_gamma = 1/(4*pi);

    // Support constants and mesh information
    SolverParameters sp;
    sp.S = 5;
    sp.R = 2*sp.S;
    sp.L = ((3 + std::sqrt(2))/2)*sp.S;
    
    sp.Nv = Nv;
    sp.Nr = Nv;
    sp.Ns = Ns;

    // Build the velocity domain as a tensor product grid
    const double dv = 2*sp.L/Nv;
    std::vector<double> v(Nv,0);

    for (int i = 0; i < Nv; ++i){
        v[i] = -sp.L + dv/2 + i*dv;
    }

    std::vector<double> vx = v;
    std::vector<double> vy = v;
    std::vector<double> vz = v;

    // Setup the BKW solution then find the corresponding collision operator Q
    const double t = 6.5;
    const double K = 1 - std::exp(-t/6);
    const double dK = std::exp(-t/6)/6;

    int grid_size = Nv*Nv*Nv;    
    std::vector<double> f_bkw(grid_size,0);
    std::vector<double> Q_bkw(grid_size,0);

    for (int i = 0; i < Nv; ++i){
        for (int j = 0; j < Nv; ++j){
            for (int k = 0; k < Nv; ++k){

                int idx = IDX3(i,j,k,Nv,Nv,Nv);
                double r_sq = vx[i]*vx[i] + vy[j]*vy[j] + vz[k]*vz[k];
                
                // Compute the BKW solution
                f_bkw[idx] = std::exp(-(r_sq)/(2*K))*((5*K-3)/K+(1-K)/(std::pow(K,2))*(r_sq));
                f_bkw[idx] *= 1/(2*std::pow(2*pi*K, 1.5));

                // Compute the derivative of f
                Q_bkw[idx] = (-3/(2*K) + r_sq/(2*std::pow(K,2)))*f_bkw[idx];
                Q_bkw[idx] += 1/(2*std::pow(2*pi*K, 1.5))*std::exp(-r_sq/(2*K))*(3/(std::pow(K,2)) + (K-2)/(std::pow(K,3))*r_sq);
                Q_bkw[idx] *= dK;

            }
        }
    }

    // Compute the quadrature rules and store their information in the solver
    SolverManager sm;
    get_gauss_legendre_rule(sp.Nr, sm.wts_gl, sm.nodes_gl, 0, sp.R);
    
    SphericalDesign sd = get_spherical_design(Ns);
    sm.sigma1_sph = sd.x;
    sm.sigma2_sph = sd.y;
    sm.sigma3_sph = sd.z;
    sm.wts_sph = std::vector<double>(sp.Ns, (4*pi)/sp.Ns);

    // Precompute and store the wave numbers for the transform as a tensor product
    std::vector<int> l;
    l.reserve(Nv);
    
    // First half: 0 to N/2 - 1
    for (int i = 0; i < Nv/2; ++i){
        l.push_back(i);
    }

    // Second half: -N/2 to -1  
    for (int i = -Nv/2; i < 0; ++i){
        l.push_back(i);
    }

    sm.l1 = l;
    sm.l2 = l;
    sm.l3 = l;
    
    // Precompute any transform weights
    precompute_weights(sm, sp, b_gamma, gamma);

    // Allocate space for the collision operator computed from the Boltzmann operator
    std::vector<double> Q(grid_size,0);

    // Container to hold the run data CPU
    std::vector<double> run_times;
    run_times.reserve(trials);

    for(int trial_idx = 0; trial_idx < trials; ++trial_idx){

        double start_time = omp_get_wtime();
        
        boltzmann_vhs_spectral_solver(Q, f_bkw, sm, sp, b_gamma, gamma); 

        double end_time = omp_get_wtime();

        // Total time returned is in seconds
        double total_time = end_time - start_time;

        // Store the time data in a vector
        run_times.push_back(total_time);

    }
    
    print_stats_summary("CPU", run_times);

    // Check the errors in the different norms and print them to the console
    double err_L1 = 0;
    double err_L2 = 0;
    double err_Linf = 0;
    double abs_diff;

    for (int idx = 0; idx < grid_size; ++idx){
        abs_diff = std::abs(Q[idx] - Q_bkw[idx]);
        err_L1 += abs_diff;
        err_L2 += abs_diff*abs_diff;
        err_Linf = std::max(err_Linf, abs_diff);
    }

    // L1 and L2 errors need to be further modified
    double dv3 = dv*dv*dv;
    err_L1 *= dv3;
    err_L2 *= dv3;
    err_L2 = std::sqrt(err_L2);

    std::cout << "Approximation errors:\n";
    std::cout << "L1 error: " << err_L1 << "\n";
    std::cout << "L2 error: " << err_L2 << "\n";
    std::cout << "Linf error: " << err_Linf << "\n\n";

    // Release the memory allocated during the precomputation phase
    precompute_release(sm);
  
    return 0;
}
