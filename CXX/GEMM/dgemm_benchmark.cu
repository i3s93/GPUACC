#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

// Argument parsing
#include <tclap/CmdLine.h>

// CUDA library
#include <cuda.h>
#include <cublas_v2.h>

// OpenMP
#include <omp.h>

// Maps a tuple for a 2D array index to a 1D index (column-major order)
// i is the row index, j is the column index, and ld is the leading dimension
// For a column-major ordering, this is the number of rows
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

extern "C" {
    void dgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* a, const int* lda,
                const double* b, const int* ldb,
                const double* beta, double* c, const int* ldc);
}

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cout << "CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
        exit(EXIT_FAILURE);
    }
}

void checkCublas(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: ";
        switch (result) {
            case CUBLAS_STATUS_NOT_INITIALIZED: std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED"; break;
            case CUBLAS_STATUS_ALLOC_FAILED: std::cerr << "CUBLAS_STATUS_ALLOC_FAILED"; break;
            case CUBLAS_STATUS_INVALID_VALUE: std::cerr << "CUBLAS_STATUS_INVALID_VALUE"; break;
            case CUBLAS_STATUS_ARCH_MISMATCH: std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH"; break;
            case CUBLAS_STATUS_MAPPING_ERROR: std::cerr << "CUBLAS_STATUS_MAPPING_ERROR"; break;
            case CUBLAS_STATUS_EXECUTION_FAILED: std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED"; break;
            case CUBLAS_STATUS_INTERNAL_ERROR: std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR"; break;
            default: std::cerr << "UNKNOWN"; break;
        }
        std::cerr << "\n";
        exit(EXIT_FAILURE);
    }
}

template<typename T>
T min(const std::vector<T> &data){
    T min_val = data[0];
    for (const auto &item: data){
        if (item < min_val) {
            min_val = item;
        }
    }
    return min_val;
}

template<typename T>
T max(const std::vector<T> &data){
    T max_val = data[0];
    for (const auto &item: data){
        if (item > max_val) {
            max_val = item;
        }
    }
    return max_val;
}

template<typename T>
T mean(const std::vector<T> &data){
    T sum = 0;
    for (const auto &item: data){
        sum += item;
    }
    return sum/static_cast<T>(data.size());
}

template<typename T>
T stdev(const std::vector<T> &data){
    T mean_val = mean(data);
    T sum_squared_diff = 0;
    for (const auto &item: data){
        sum_squared_diff += std::pow(item - mean_val,2);
    }
    T denom = static_cast<T>(std::max(1, static_cast<int>(data.size() - 1)));
    return std::sqrt(sum_squared_diff / denom);
}

template<typename T>
void print_stats_summary(const std::string &device_name, const std::vector<T> &data){
    std::cout << "\nRun statistics for " << device_name << std::endl;
    std::cout << "Total number of samples taken: " << data.size() << std::endl;
    std::cout << std::scientific << std::setprecision(4) << "Mean runtime (ms): " << mean(data) << "\n";
    std::cout << std::scientific << std::setprecision(4) << "Min runtime (ms): " << min(data) << "\n";
    std::cout << std::scientific << std::setprecision(4) << "Max runtime (ms): " << max(data) << "\n";
    std::cout << std::scientific << std::setprecision(4) << "stdev: " << stdev(data) << "\n";
    std::cout << "\n";
    return;
}

int main(int argc, char** argv) {

    int M, N, K, trials;

    try {
        // Create each of the arguments
        TCLAP::CmdLine cmd("Command description message", ' ', "1.0");
        TCLAP::ValueArg<int> M_Arg("M", "M_size", "Number of rows of A and C", false, 1024, "int");
        TCLAP::ValueArg<int> N_Arg("N", "N_size", "Number of columns of B and C", false, 1024, "int");
        TCLAP::ValueArg<int> K_Arg("K", "K_size", "Number of columns of A and rows of B", false, 1024, "int");
        TCLAP::ValueArg<int> t_Arg("t", "trials", "Number of trials to use for statistics", false, 100, "int");

        cmd.add(M_Arg);
        cmd.add(N_Arg);
        cmd.add(K_Arg);
        cmd.add(t_Arg);

        // Parse the argv array.
        cmd.parse(argc, argv);

        // Assign parsed values to variables
        M = M_Arg.getValue();
        N = N_Arg.getValue();
        K = K_Arg.getValue();
        trials = t_Arg.getValue();

        std::cout << "\nRun arguments:" << "\n";
        std::cout << "M = " << M << "\n";
        std::cout << "N = " << N << "\n";
        std::cout << "K = " << K << "\n";
        std::cout << "trials = " << trials << "\n";
    } catch (TCLAP::ArgException &e)
    { std::cerr << "error: " << e.error() << " for arg " << e.argId() << "\n"; }

    // Initialize cuBLAS
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    // Allocate and initialize host matrices
    std::vector<double> A_h(M*K);
    std::vector<double> B_h(K*N);
    std::vector<double> C_h(M*N);

    // Setup the random number generator on the host
    std::mt19937 generator(123);
    std::normal_distribution<double> distribution(0.0, 1.0);

    // Fill A and B with random data
    for (int j = 0; j < K; ++j) {
        for (int i = 0; i < M; ++i) {
            A_h[IDX2C(i, j, M)] = distribution(generator);
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < K; ++i) {
            B_h[IDX2C(i, j, K)] = distribution(generator);
        }
    }

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            C_h[IDX2C(i, j, M)] = 0;
        }
    }

    // Allocate device arrays
    double *A_d, *B_d, *C_d;
    checkCuda(cudaMalloc((void **)&A_d, M*K*sizeof(double)));
    checkCuda(cudaMalloc((void **)&B_d, K*N*sizeof(double)));
    checkCuda(cudaMalloc((void **)&C_d, M*N*sizeof(double)));

    // Copy matrices from host to device
    checkCuda(cudaMemcpy(A_d, A_h.data(), M*K*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(B_d, B_h.data(), K*N*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(C_d, C_h.data(), M*N*sizeof(double), cudaMemcpyHostToDevice));

    // Coefficients for the gemm: C = alpha*A*B + beta*C
    double alpha = 1;
    double beta = 0;

    cublasOperation_t trans_A = CUBLAS_OP_N;
    cublasOperation_t trans_B = CUBLAS_OP_N;

    // Measure time for the matrix multiplication using CUDA events
    cudaEvent_t dgemm_start, dgemm_stop;
    cudaEventCreate(&dgemm_start);
    cudaEventCreate(&dgemm_stop);

    // This needs to be a float (required by CUDA)
    float dgemm_total_time;

    // Container to hold the run data GPU
    std::vector<float> dev_times;
    
    for(int trial_idx = 0; trial_idx < trials; ++trial_idx){

        // Record the start time
        cudaEventRecord(dgemm_start);
        checkCublas(cublasDgemm(handle, trans_A, trans_B, M, N, K, &alpha, A_d, M, B_d, K, &beta, C_d, M));
        cudaEventRecord(dgemm_stop);

        // Ensure that the host waits until the device kernel completes
        cudaEventSynchronize(dgemm_stop);

        // Stop the timer and print the elapsed time in milliseconds
        cudaEventElapsedTime(&dgemm_total_time, dgemm_start, dgemm_stop);

        // Save the timing data for later
        dev_times.push_back(dgemm_total_time);
    }

    print_stats_summary("GPU", dev_times);

    // Copy result from device to host to check for correctness (later)
    checkCuda(cudaMemcpy(C_h.data(), C_d, M*N*sizeof(double), cudaMemcpyDeviceToHost));

    // Variable to hold the output for the host result obtained with BLAS
    std::vector<double> C_exact(M*N, 0);

    // Compute the solution on the host side using BLAS
    // We reuse the coefficients and leading dimensions in the call
    char transa = 'N';
    char transb = 'N';

    // Container to hold the run data CPU
    std::vector<double> host_times;

    for(int trial_idx = 0; trial_idx < trials; ++trial_idx){

        double host_start_time = omp_get_wtime();
        dgemm_(&transa, &transb, &M, &N, &K, &alpha, A_h.data(), &M, B_h.data(), &N, &beta, C_exact.data(), &K);
        double host_end_time = omp_get_wtime();

        // Total time returned is in seconds, so we convert it to ms
        double host_dgemm_total_time = host_end_time - host_start_time;
        host_dgemm_total_time *= 1000;

        // Store the time data in a vector
        host_times.push_back(host_dgemm_total_time);
    }
    
    print_stats_summary("CPU", host_times);

    double max_error = 0;

    #pragma omp parallel for collapse(2) reduction(max:max_error)
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            max_error = std::max(max_error, std::abs(C_h[IDX2C(i, j, M)] - C_exact[IDX2C(i, j, M)]));
        }
    }

    std::cout << "dgemm max error (between host and device): " << max_error << "\n";
    std::cout << "\n";

    // Clean up the device arrays, events, and the cuBLAS handle
    checkCuda(cudaFree(A_d));
    checkCuda(cudaFree(B_d));
    checkCuda(cudaFree(C_d));

    cudaEventDestroy(dgemm_start);
    cudaEventDestroy(dgemm_stop);

    checkCublas(cublasDestroy(handle));

    return 0;
}
