#include "boltzmann_collisions_omp.hpp"

// Custom reduction for complex types since OpenMP does not provide this
#pragma omp declare	\
reduction(	\
	+ : \
	std::complex<double> :	\
	omp_out += omp_in )	\
initializer( omp_priv = omp_orig )

void precompute_weights_omp(SolverManager &sm, SolverParameters &sp, 
                        const double b_gamma, const double gamma){

    // Unpack the parameters for the evaluation
    int Nv = sp.Nv;
    int Nr = sp.Nr;
    int Ns = sp.Ns;
    double L = sp.L;

    // Grid sizes for loops and normalizations
    int batch_size = Nr*Ns; 
    int grid_size = Nv*Nv*Nv;

    // Unpack the quadrature data
    const std::vector<double>& wts_gl = sm.wts_gl;
    const std::vector<double>& nodes_gl = sm.nodes_gl;
    
    const std::vector<double>& sigma1 = sm.sigma1_sph;
    const std::vector<double>& sigma2 = sm.sigma2_sph;
    const std::vector<double>& sigma3 = sm.sigma3_sph;

    // Extract the wave number vectors
    const std::vector<int>& l1 = sm.l1;
    const std::vector<int>& l2 = sm.l2;
    const std::vector<int>& l3 = sm.l3;

    // Allocate space for the weights
    // Note: alpha2 = conj(alpha1) so we don't store it
    sm.alpha1 = (std::complex<double>*)fftw_malloc(batch_size*grid_size*sizeof(std::complex<double>)); 
    sm.beta1 = (double*)fftw_malloc(batch_size*grid_size*sizeof(double));
    sm.beta2 = (double*)fftw_malloc(grid_size*sizeof(double));

    #pragma omp parallel
    {

    // Compute the weights alpha1, beta1, and beta2 for the transforms
    #pragma omp for collapse(4)
    for (int r = 0; r < Nr; ++r){
       for (int s = 0; s < Ns; ++s){
           for (int i = 0; i < Nv; ++i){
               for (int j = 0; j < Nv; ++j){
                    #pragma omp simd
                    for (int k = 0; k < Nv; ++k){

                        int idx5 = IDX5(r,s,i,j,k,Nr,Ns,Nv,Nv,Nv);
                        double l_dot_sigma = l1[i]*sigma1[s] + l2[j]*sigma2[s] + l3[k]*sigma3[s];
                        double norm_l = std::sqrt(l1[i]*l1[i] + l2[j]*l2[j] + l3[k]*l3[k]);
                        sm.alpha1[idx5] = std::exp(std::complex<double>(0,-(pi/(2*L))*nodes_gl[r]*l_dot_sigma));
                        sm.beta1[idx5] = 4*pi*b_gamma*sincc(pi*nodes_gl[r]*norm_l/(2*L));

                    }
                }
            }
        }
    }

    // Initialize the beta2 to zero, since this requires an accumulation
    #pragma omp for collapse(2)
    for (int i = 0; i < Nv; ++i){
        for (int j = 0; j < Nv; ++j){
            #pragma omp simd
            for (int k = 0; k < Nv; ++k){

                int idx3 = IDX3(i,j,k,Nv,Nv,Nv);
                sm.beta2[idx3] = 0;

            }
        }
    }

    double* beta2 = sm.beta2;

    #pragma omp for collapse(3) reduction(+:beta2[:grid_size])
    for (int r = 0; r < Nr; ++r){
        for (int i = 0; i < Nv; ++i){
            for (int j = 0; j < Nv; ++j){
                #pragma omp simd
                for (int k = 0; k < Nv; ++k){

                    int idx3 = IDX3(i,j,k,Nv,Nv,Nv);
                    double norm_l = std::sqrt(l1[i]*l1[i] + l2[j]*l2[j] + l3[k]*l3[k]);
                    beta2[idx3] += 16*pi*pi*b_gamma*wts_gl[r]*std::pow(nodes_gl[r], gamma+2)*sincc(pi*nodes_gl[r]*norm_l/L);

                }
            }
        }
    }

    } // End of parallel region

    return;
}

void precompute_release_omp(SolverManager &sm){

    // Release the memory from the precomputation phase
    fftw_free(sm.alpha1);
    fftw_free(sm.beta1);
    fftw_free(sm.beta2);
    
    return;
}



void boltzmann_vhs_spectral_solver_omp(std::vector<double> &Q,
                                       std::vector<double> &f_in,
                                       const SolverManager &sm, const SolverParameters &sp,
                                       const double b_gamma, const double gamma){

    // Initialize the threading environment and set the number of threads to be used in the planning phase
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_num_threads());

    // Unpack the parameters for the evaluation
    int Nv = sp.Nv;
    int Nr = sp.Nr;
    int Ns = sp.Ns;
    double L = sp.L;

    // Unpack the quadrature data
    const std::vector<double>& wts_gl = sm.wts_gl;
    const std::vector<double>& nodes_gl = sm.nodes_gl;

    const std::vector<double>& wts_sph = sm.wts_sph;
    const std::vector<double>& sigma1 = sm.sigma1_sph;
    const std::vector<double>& sigma2 = sm.sigma2_sph;
    const std::vector<double>& sigma3 = sm.sigma3_sph;

    // Extract the wave number vectors
    const std::vector<int>& l1 = sm.l1;
    const std::vector<int>& l2 = sm.l2;
    const std::vector<int>& l3 = sm.l3;

    // Grid sizes for loops and normalizations
    int batch_size = Nr*Ns; 
    int grid_size = Nv*Nv*Nv;

    // Allocations for the various transforms involved (including forward and backward)
    std::complex<double> *f = (std::complex<double>*)fftw_malloc(grid_size*sizeof(std::complex<double>));
    std::complex<double> *f_hat = (std::complex<double>*)fftw_malloc(grid_size*sizeof(std::complex<double>));
        
    std::complex<double> *alpha1_times_f = (std::complex<double>*)fftw_malloc(batch_size*grid_size*sizeof(std::complex<double>));
    std::complex<double> *alpha1_times_f_hat = (std::complex<double>*)fftw_malloc(batch_size*grid_size*sizeof(std::complex<double>));

    std::complex<double> *alpha2_times_f = (std::complex<double>*)fftw_malloc(batch_size*grid_size*sizeof(std::complex<double>));
    std::complex<double> *alpha2_times_f_hat = (std::complex<double>*)fftw_malloc(batch_size*grid_size*sizeof(std::complex<double>));
    
    std::complex<double> *transform_prod = (std::complex<double>*)fftw_malloc(batch_size*grid_size*sizeof(std::complex<double>));
    std::complex<double> *transform_prod_hat = (std::complex<double>*)fftw_malloc(batch_size*grid_size*sizeof(std::complex<double>));

    std::complex<double> *Q_gain = (std::complex<double>*)fftw_malloc(grid_size*sizeof(std::complex<double>));
    std::complex<double> *Q_gain_hat = (std::complex<double>*)fftw_malloc(grid_size*sizeof(std::complex<double>));

    std::complex<double> *beta2_times_f = (std::complex<double>*)fftw_malloc(grid_size*sizeof(std::complex<double>));
    std::complex<double> *beta2_times_f_hat = (std::complex<double>*)fftw_malloc(grid_size*sizeof(std::complex<double>));
    
    // Since FFTW is not normalized, we need to compute the scaling applied in the inverse FFT
    // Be careful to avoid integer arithmetic
    double fft_scale = 1.0/grid_size;

    std::string fname("wisdom_omp.dat");

    if(fftw_import_wisdom_from_filename(fname.c_str()) == 0){
        std::cout << "Failed to import wisdom from file: " << fname << "\n";
    }

    // Creating plans for each of the transforms 
    // Some of these are batched FFTs
    fftw_plan fft_f = fftw_plan_dft_3d(Nv, Nv, Nv, 
                                       reinterpret_cast<fftw_complex*>(f), 
                                       reinterpret_cast<fftw_complex*>(f_hat), 
                                       FFTW_FORWARD, FFTW_ESTIMATE);

    int batched_rank = 3; // Each FFT is applied to a three-dimensional row-major array
    int batched_dims[] = {Nv, Nv, Nv}; // Dimensions of the arrays used in each transform
    int idist = Nv*Nv*Nv; // Each array is separated by idist elements
    int odist = idist; // Each array is separated by odist elements
    int istride = 1; // Arrays are contiguous in memory
    int ostride = 1; // Arrays are contiguous in memory
    int *inembed = batched_dims; // The array is not embedded in a larger array
    int *onembed = batched_dims; // The array is not embedded in a larger array

    fftw_plan ifft_alpha1_times_f_hat = fftw_plan_many_dft(batched_rank, batched_dims, batch_size,
                                                           reinterpret_cast<fftw_complex*>(alpha1_times_f_hat), 
                                                           inembed, istride, idist,
                                                           reinterpret_cast<fftw_complex*>(alpha1_times_f),
                                                           onembed, ostride, odist,
                                                           FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_plan ifft_alpha2_times_f_hat = fftw_plan_many_dft(batched_rank, batched_dims, batch_size,
                                                           reinterpret_cast<fftw_complex*>(alpha2_times_f_hat), 
                                                           inembed, istride, idist,
                                                           reinterpret_cast<fftw_complex*>(alpha2_times_f),
                                                           onembed, ostride, odist,
                                                           FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_plan fft_product = fftw_plan_many_dft(batched_rank, batched_dims, batch_size,
                                               reinterpret_cast<fftw_complex*>(transform_prod), 
                                               inembed, istride, idist,
                                               reinterpret_cast<fftw_complex*>(transform_prod_hat),
                                               onembed, ostride, odist,
                                               FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_plan ifft_Q_gain_hat = fftw_plan_dft_3d(Nv, Nv, Nv, 
                                                 reinterpret_cast<fftw_complex*>(Q_gain_hat), 
                                                 reinterpret_cast<fftw_complex*>(Q_gain), 
                                                 FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_plan ifft_beta2_times_f_hat = fftw_plan_dft_3d(Nv, Nv, Nv, 
                                                        reinterpret_cast<fftw_complex*>(beta2_times_f_hat), 
                                                        reinterpret_cast<fftw_complex*>(beta2_times_f), 
                                                        FFTW_BACKWARD, FFTW_ESTIMATE); 

    // Export wisdom immediately after plan creation
    fftw_export_wisdom_to_filename(fname.c_str());

    #pragma omp parallel
    {

    #pragma omp for collapse(2)
    for (int i = 0; i < Nv; ++i){
        for (int j = 0; j < Nv; ++j){
            #pragma omp simd
            for (int k = 0; k < Nv; ++k){
                int idx3 = IDX3(i,j,k,Nv,Nv,Nv);    
                f[idx3] = f_in[idx3];
                Q_gain_hat[idx3] = 0;
            }
        }
    }

    #pragma omp barrier

    #pragma omp single
    fftw_execute(fft_f); // Transform f to get f_hat
     
    // Compute the products alpha1*f_hat and alpha2*f_hat
    // Transforms are properly normalized in this step
    #pragma omp for collapse(4)
    for (int r = 0; r < Nr; ++r){
       for (int s = 0; s < Ns; ++s){
           for (int i = 0; i < Nv; ++i){
               for (int j = 0; j < Nv; ++j){
                    #pragma omp simd
                    for (int k = 0; k < Nv; ++k){

                        int idx3 = IDX3(i,j,k,Nv,Nv,Nv);
                        int idx5 = IDX5(r,s,i,j,k,Nr,Ns,Nv,Nv,Nv);
                        alpha1_times_f_hat[idx5] = fft_scale*sm.alpha1[idx5]*f_hat[idx3]; 
                        alpha2_times_f_hat[idx5] = fft_scale*std::conj(sm.alpha1[idx5])*f_hat[idx3];

                    }
                }
            }
        }
    }

    #pragma omp barrier

    // Apply batched iFFTs to alpha1_times_f_hat, alpha2_times_f_hat
    #pragma omp single
    {
    fftw_execute(ifft_alpha1_times_f_hat);
    fftw_execute(ifft_alpha2_times_f_hat);
    }

    // Compute the product of the transforms in physical space
    #pragma omp for collapse(4)
    for (int r = 0; r < Nr; ++r){
       for (int s = 0; s < Ns; ++s){
           for (int i = 0; i < Nv; ++i){
               for (int j = 0; j < Nv; ++j){
                    #pragma omp simd
                    for (int k = 0; k < Nv; ++k){

                        int idx5 = IDX5(r,s,i,j,k,Nr,Ns,Nv,Nv,Nv);
                        transform_prod[idx5] = alpha1_times_f[idx5]*alpha2_times_f[idx5];

                    }
                }
            }
        }
    }    

    #pragma omp barrier

    // Apply a batched FFT to the product
    #pragma omp single
    fftw_execute(fft_product);

    // Update the gain term in the frequency domain
    // Each term that is added is normalized 
    #pragma omp for collapse(4) reduction(+:Q_gain_hat[:grid_size])
    for (int r = 0; r < Nr; ++r){
       for (int s = 0; s < Ns; ++s){
           for (int i = 0; i < Nv; ++i){
               for (int j = 0; j < Nv; ++j){
                    #pragma omp simd
                    for (int k = 0; k < Nv; ++k){

                        int idx3 = IDX3(i,j,k,Nv,Nv,Nv);
                        int idx5 = IDX5(r,s,i,j,k,Nr,Ns,Nv,Nv,Nv);
                        Q_gain_hat[idx3] += fft_scale*wts_gl[r]*wts_sph[s]*std::pow(nodes_gl[r],gamma+2)*sm.beta1[idx5]*transform_prod_hat[idx5]; 

                    }
                }
            }
        }
    }

    // Apply the weights beta2 to f_hat and normalize    
    #pragma omp for collapse(2)
    for (int i = 0; i < Nv; ++i){
        for (int j = 0; j < Nv; ++j){
            #pragma omp simd
            for (int k = 0; k < Nv; ++k){

                int idx3 = IDX3(i,j,k,Nv,Nv,Nv);
                beta2_times_f_hat[idx3] = fft_scale*sm.beta2[idx3]*f_hat[idx3];

            }
        }
    }

    #pragma omp barrier
    
    // Transform Q_gain back to physical space
    #pragma omp single 
    fftw_execute(ifft_Q_gain_hat);

    // Transform beta2_times_f_hat back to physical space
    #pragma omp single
    fftw_execute(ifft_beta2_times_f_hat);

    // Compute the final form for Q = real(Q_gain - Q_loss)
    #pragma omp for collapse(2)
    for (int i = 0; i < Nv; ++i){
        for (int j = 0; j < Nv; ++j){
            #pragma omp simd
            for (int k = 0; k < Nv; ++k){

                int idx3 = IDX3(i,j,k,Nv,Nv,Nv);
                std::complex<double> Q_loss_ijk = beta2_times_f[idx3]*f[idx3];
                Q[idx3] = Q_gain[idx3].real() - Q_loss_ijk.real();

            }
        }            
    }

    } // End of omp parallel region (implicit barrier)

    // Destory the plans for each of the transforms
    fftw_destroy_plan(fft_f);
    fftw_destroy_plan(ifft_alpha1_times_f_hat);
    fftw_destroy_plan(ifft_alpha2_times_f_hat);
    fftw_destroy_plan(fft_product);
    fftw_destroy_plan(ifft_Q_gain_hat);
    fftw_destroy_plan(ifft_beta2_times_f_hat);

    // Free any allocated memory
    fftw_free(f);
    fftw_free(f_hat);
    fftw_free(alpha1_times_f);
    fftw_free(alpha1_times_f_hat);
    fftw_free(alpha2_times_f);
    fftw_free(alpha2_times_f_hat);
    fftw_free(transform_prod);
    fftw_free(transform_prod_hat);
    fftw_free(Q_gain);
    fftw_free(Q_gain_hat);
    fftw_free(beta2_times_f);
    fftw_free(beta2_times_f_hat);

    return;
}
