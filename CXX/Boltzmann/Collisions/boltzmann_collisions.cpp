#include "boltzmann_collisions.hpp"

void boltzmann_vhs_spectral_solver(std::vector<double> &Q,
                                   std::vector<double> &f_in,
                                   const SolverManager &sm, const SolverParameters &sp,
                                   const double b_gamma, const double gamma){

// Unpack the parameters for the evaluation
int Nv = sp.Nv;
int Nr = sp.Nr;
int Ns = sp.Ns;
double L = sp.L;

// Unpack the quadrature data
std::vector<double> wts_gl = sm.wts_gl;
std::vector<double> nodes_gl = sm.nodes_gl;

std::vector<double> wts_sph = sm.wts_sph;
std::vector<double> sigma1 = sm.sigma1_sph;
std::vector<double> sigma2 = sm.sigma2_sph;
std::vector<double> sigma3 = sm.sigma3_sph;

// Extract the wave number vectors
std::vector<int> l1 = sm.l1;
std::vector<int> l2 = sm.l2;
std::vector<int> l3 = sm.l3;

// Temporary for weights used to compute the loss term
std::vector<std::complex<double>> beta1 = std::vector<std::complex<double>>(Nv*Nv*Nv);
std::vector<std::complex<double>> beta2 = std::vector<std::complex<double>>(Nv*Nv*Nv);

// Allocations for the various transforms involved (including forward and backward)
// These can be optimized with in-place transforms
std::vector<std::complex<double>> f_hat = std::vector<std::complex<double>>(Nv*Nv*Nv);

std::vector<std::complex<double>> alpha1_times_f = std::vector<std::complex<double>>(Nv*Nv*Nv);
std::vector<std::complex<double>> alpha1_times_f_hat = std::vector<std::complex<double>>(Nv*Nv*Nv);

std::vector<std::complex<double>> alpha2_times_f = std::vector<std::complex<double>>(Nv*Nv*Nv);
std::vector<std::complex<double>> alpha2_times_f_hat = std::vector<std::complex<double>>(Nv*Nv*Nv);

std::vector<std::complex<double>> transform_prod = std::vector<std::complex<double>>(Nv*Nv*Nv);
std::vector<std::complex<double>> transform_prod_hat = std::vector<std::complex<double>>(Nv*Nv*Nv);

std::vector<std::complex<double>> Q_gain = std::vector<std::complex<double>>(Nv*Nv*Nv, 0);
std::vector<std::complex<double>> Q_gain_hat = std::vector<std::complex<double>>(Nv*Nv*Nv, 0);

std::vector<std::complex<double>> beta2_times_f = std::vector<std::complex<double>>(Nv*Nv*Nv, 0);
std::vector<std::complex<double>> beta2_times_f_hat = std::vector<std::complex<double>>(Nv*Nv*Nv);

// Convert the input distribution from double to complex
// TO-DO: Once this works, convert things to r2c and c2r transforms
std::vector<std::complex<double>> f(Nv*Nv*Nv);

for (int i = 0; i < Nv; ++i){
    for (int j = 0; j < Nv; ++j){
        for (int k = 0; k < Nv; ++k){
            f[IDX(i,j,k,Nv,Nv)] = f_in[IDX(i,j,k,Nv,Nv)];
        }
    }
}

// if(fftw_import_wisdom_from_filename(fname.c_str()) == 0){
//     std::cout << "Failed to import wisdom from file: " << fname << "\n";
// }

// Creating plans for each of the transforms 
// Start with FFTW_ESTIMATE, but use FFTW_PATIENT for faster transforms
// Note: planning functions are not thread safe
fftw_plan fft_f = fftw_plan_dft_3d(Nv, Nv, Nv, 
                                   reinterpret_cast<fftw_complex*>(f.data()), 
                                   reinterpret_cast<fftw_complex*>(f_hat.data()),
                                   FFTW_FORWARD, FFTW_ESTIMATE);

fftw_plan ifft_alpha1_times_f_hat = fftw_plan_dft_3d(Nv, Nv, Nv, 
                                                    reinterpret_cast<fftw_complex*>(alpha1_times_f_hat.data()), 
                                                    reinterpret_cast<fftw_complex*>(alpha1_times_f.data()),
                                                    FFTW_BACKWARD, FFTW_ESTIMATE);

fftw_plan ifft_alpha2_times_f_hat = fftw_plan_dft_3d(Nv, Nv, Nv, 
                                                    reinterpret_cast<fftw_complex*>(alpha2_times_f_hat.data()), 
                                                    reinterpret_cast<fftw_complex*>(alpha2_times_f.data()),
                                                    FFTW_BACKWARD, FFTW_ESTIMATE);

fftw_plan fft_product = fftw_plan_dft_3d(Nv, Nv, Nv, 
                                        reinterpret_cast<fftw_complex*>(transform_prod.data()), 
                                        reinterpret_cast<fftw_complex*>(transform_prod_hat.data()),
                                        FFTW_FORWARD, FFTW_ESTIMATE);

fftw_plan ifft_Q_gain_hat = fftw_plan_dft_3d(Nv, Nv, Nv,
                                            reinterpret_cast<fftw_complex*>(Q_gain_hat.data()),
                                            reinterpret_cast<fftw_complex*>(Q_gain.data()),
                                            FFTW_BACKWARD, FFTW_ESTIMATE);

fftw_plan ifft_Q_loss_hat = fftw_plan_dft_3d(Nv, Nv, Nv, 
                                            reinterpret_cast<fftw_complex*>(beta2_times_f_hat.data()), 
                                            reinterpret_cast<fftw_complex*>(beta2_times_f.data()),
                                            FFTW_BACKWARD, FFTW_ESTIMATE);

// Transform f to get f_hat
fftw_execute(fft_f);

for (int r = 0; r < Nr; ++r){
    for (int s = 0; s < Ns; ++s){

        // Compute the complex weights alpha1 and alpha2 to scale the gain term
        for (int i = 0; i < Nv; ++i){
            for (int j = 0; j < Nv; ++j){
                for (int k = 0; k < Nv; ++k){
                    double l_dot_sigma = l1[i]*sigma1[s] + l2[j]*sigma2[s] + l3[k]*sigma3[s];
                    std::complex<double> alpha1 = std::exp(std::complex<double>(0,-(pi/(2*L))*nodes_gl[r]*l_dot_sigma));
                    std::complex<double> alpha2 = std::conj(alpha1);
                    double norm_l = std::sqrt(std::pow(l1[i],2) + std::pow(l2[j],2) + std::pow(l3[k],2));
                    
                    alpha1_times_f_hat[IDX(i,j,k,Nv,Nv)] = alpha1*f_hat[IDX(i,j,k,Nv,Nv)];
                    alpha2_times_f_hat[IDX(i,j,k,Nv,Nv)] = alpha2*f_hat[IDX(i,j,k,Nv,Nv)];
                    beta1[IDX(i,j,k,Nv,Nv)] = 4*pi*b_gamma*sincc(pi*nodes_gl[r]*norm_l/(2*L));
                }
            }
        }

        // Invert the weighted transforms of f
        // These are not normalized, so we need to divide by the number of elements later
        fftw_execute(ifft_alpha1_times_f_hat);
        fftw_execute(ifft_alpha2_times_f_hat);

        // Compute the product of the transforms in physical space
        for (int i = 0; i < Nv; ++i){
            for (int j = 0; j < Nv; ++j){
                for (int k = 0; k < Nv; ++k){
                    double normalization = 1/(Nv*Nv*Nv);
                    transform_prod[IDX(i,j,k,Nv,Nv)] = alpha1_times_f[IDX(i,j,k,Nv,Nv)]*alpha2_times_f[IDX(i,j,k,Nv,Nv)];
                    transform_prod[IDX(i,j,k,Nv,Nv)] *= std::pow(normalization,2);
                }
            }
        }

        // Transform the product back to the frequency domain
        fftw_execute(fft_product);

        // Update the gain term in the frequency domain 
        for (int i = 0; i < Nv; ++i){
            for (int j = 0; j < Nv; ++j){
                for (int k = 0; k < Nv; ++k){
                    Q_gain[IDX(i,j,k,Nv,Nv)] += wts_gl[r]*wts_sph[s]*std::pow(nodes_gl[r],gamma+2)*beta1[IDX(i,j,k,Nv,Nv)]*transform_prod[IDX(i,j,k,Nv,Nv)];
                }
            }
        }
    }

    // Compute the complex weights beta2
    for (int i = 0; i < Nv; ++i){
        for (int j = 0; j < Nv; ++j){
            for (int k = 0; k < Nv; ++k){
                double norm_l = std::sqrt(std::pow(l1[i],2) + std::pow(l2[j],2) + std::pow(l3[k],2));
                beta2[IDX(i,j,k,Nv,Nv)] += 16*std::pow(pi,2)*b_gamma*wts_gl[r]*std::pow(nodes_gl[r], gamma+2)*sincc(pi*nodes_gl[r]*norm_l/L);
            }
        }
    }

}

// Apply weights beta2 to f_hat
for (int i = 0; i < Nv; ++i){
    for (int j = 0; j < Nv; ++j){
        for (int k = 0; k < Nv; ++k){
            beta2_times_f_hat[IDX(i,j,k,Nv,Nv)] = beta2[IDX(i,j,k,Nv,Nv)]*f_hat[IDX(i,j,k,Nv,Nv)];
        }
    }
}

// Transform Q_gain back to physical space
// This needs to be normalized
fftw_execute(ifft_Q_gain_hat);

// Transform beta2_times_f_hat back to physical space
// This also needs to be normalized
fftw_execute(ifft_Q_loss_hat);

// Compute the final form for Q = real(Q_gain - Q_loss)
// We include the above normalizations here as we build Q
for (int i = 0; i < Nv; ++i){
    for (int j = 0; j < Nv; ++j){
        for (int k = 0; k < Nv; ++k){
            double normalization = 1/(Nv*Nv*Nv);
            std::complex<double> Q_loss = normalization*beta2_times_f[IDX(i,j,k,Nv,Nv)]*f[IDX(i,j,k,Nv,Nv)];
            Q_gain[IDX(i,j,k,Nv,Nv)] *= normalization;
            Q[IDX(i,j,k,Nv,Nv)] = Q_gain[IDX(i,j,k,Nv,Nv)].real() - Q_loss.real();
        }
    }
}

// Destory the plans for each of the transforms
fftw_destroy_plan(fft_f);
fftw_destroy_plan(ifft_alpha1_times_f_hat);
fftw_destroy_plan(ifft_alpha2_times_f_hat);
fftw_destroy_plan(fft_product);
fftw_destroy_plan(ifft_Q_gain_hat);
fftw_destroy_plan(ifft_Q_loss_hat);

}
