#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <cuda.h>
#include <cublas_v2.h>
#include <omp.h>

extern "C" {
    void dgemm_(const char* transa, const char* transb,
                const int* m, const int* n, const int* k,
                const double* alpha, const double* a, const int* lda,
                const double* b, const int* ldb,
                const double* beta, double* c, const int* ldc);
}

// Maps a tuple for a 2D array index to a 1D index (column-major order)
// i is the row index, j is the column index, and ld is the leading dimension
// For a column-major ordering, this is the number of rows
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cout << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
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
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Initialize cuBLAS
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    // Matrix dimensions (make these command line arguments)
    int M = 1024; // Number of rows of A and C
    int N = 1024; // Number of columns of B and C
    int K = 1024; // Number of columns of A and rows of B

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
    
    // Record the start time
    cudaEventRecord(dgemm_start);

    checkCublas(cublasDgemm(handle, trans_A, trans_B, M, N, K, &alpha, A_d, M, B_d, K, &beta, C_d, M));

    cudaEventRecord(dgemm_stop);

    // Ensure that the host waits until the device kernel completes
    cudaEventSynchronize(dgemm_stop);

    // Stop the timer and print the elapsed time in milliseconds
    float dgemm_total_time = 0; // This needs to be a float
    cudaEventElapsedTime(&dgemm_total_time, dgemm_start, dgemm_stop);

    std::cout << std::scientific << std::setprecision(6) "cuBLAS dgemm total time (ms): " << dgemm_total_time << std::endl;

    // Copy result from device to host to check for correctness
    checkCuda(cudaMemcpy(C_h.data(), C_d, M*N*sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> C_exact(M*N, 0);

    // Compute the solution on the host side using BLAS
    // We reuse the coefficients and leading dimensions in the call
    char transa = 'N';
    char transb = 'N';

    double host_start_time = omp_get_wtime();

    dgemm_(&transa, &transb, &M, &N, &K, &alpha, A_h.data(), &M, B_h.data(), &N, &beta, C_exact.data(), &K);

    double host_end_time = omp_get_wtime();

    // Total time returned is in seconds, so we convert it to ms
    double host_dgemm_total_time = host_end_time - host_start_time;
    host_total_time *= 1000;

    std::cout << std::scientific << std::setprecision(6) "BLAS dgemm total time (ms): " << host_dgemm_total_time << std::endl;

    // #pragma omp parallel for collapse(2)
    // for (int j = 0; j < N; ++j) {
    //     for (int k = 0; k < K; ++k) {
    //         for (int i = 0; i < M; ++i) {
    //             C_exact[IDX2C(i, j, M)] += A_h[IDX2C(i, k, M)] * B_h[IDX2C(k, j, K)];
    //         }
    //     }
    // }

    double max_error = 0;

    #pragma omp parallel for collapse(2) reduction(max:max_error)
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            max_error = std::max(max_error, std::abs(C_h[IDX2C(i, j, M)] - C_exact[IDX2C(i, j, M)]));
        }
    }

    std::cout << "cuBLAS dgemm max error: " << max_error << std::endl;

    // Clean up the device arrays, events, and the cuBLAS handle
    checkCuda(cudaFree(A_d));
    checkCuda(cudaFree(B_d));
    checkCuda(cudaFree(C_d));

    cudaEventDestroy(dgemm_start);
    cudaEventDestroy(dgemm_stop);

    checkCublas(cublasDestroy(handle));

    return 0;
}
