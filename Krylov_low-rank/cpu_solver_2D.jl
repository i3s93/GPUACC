# This script tests a CPU implementation of a Sylvester solver
# based on Krylov subspace methods.

using ArgParse

# Retrieve the command line arguments
settings = ArgParseSettings()

@add_arg_table settings begin
    "--Lx"
        help = "Domain length along x";
        arg_type = Float64
        default = 1.0
    "--Ly"
        help = "Domain length along y";
        arg_type = Float64
        default = 1.0
    "--Nx"
        help = "Number grid points in x";
        arg_type = Int
        default = 101
    "--Ny"
        help = "Number grid points in y";
        arg_type = Int
        default = 101
    "--rel_eps"
        help = "Relative truncation tolerance for SVD truncation"
        arg_type = Float64
        default = 1.0e-3
    "--use_mkl"
        help = "Use the Intel Math Kernel Library rather than OpenBLAS"
        arg_type = Bool
        default = false
end

# Parse the arguments and print them to the command line
parsed_args = parse_args(settings)

println("Options used:")
for (arg,val) in parsed_args
    println("  $arg  =  $val")
end

# Get the individual command line arguments from the dictionary
Lx = parsed_args["Lx"]
Ly = parsed_args["Ly"]
Nx = parsed_args["Nx"]
Ny = parsed_args["Ny"]
rel_eps = parsed_args["rel_eps"]
use_mkl = parsed_args["use_mkl"]

import Base: find_package

# Check if MKL is available. If so, use it.
if use_mkl && find_package("MKL") !== nothing
	println("Intel MKL installation found.")
	using MKL
else
	println("Running with the default OpenBLAS installation.")
end

using Printf
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using MatrixEquations

using InteractiveUtils
using Profile
using ProfileView

include("ndgrid.jl")
include("extended_krylov_step.jl")

# Get the number of BLAS threads and check the configuration
println("Number of BLAS threads:", BLAS.get_num_threads())
println("BLAS config:", BLAS.get_config())

# Setup the domain as well as the differentiation matrices
# Exclude the first and last endpoints for the boundary conditions
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
x = [i*dx for i in 0:(Nx-2)]
y = [j*dy for j in 0:(Ny-2)]
X, Y = ndgrid(x, y)

# Define most of Dxx by its diagonals.
d0 =  fill(-2/dx^2, Nx-1)    # main diagonal
dpm = ones(Nx-2)/dx^2       # super- and subdiagonal
Dxx = spdiagm(-1=>dpm, 0=>d0, 1=>dpm)
Dyy = Dxx

# Define matrices for the implicit scheme
# Can either build the matrices in a full or sparse manner
dtn = 1.0e-2           # Time step size
d1 = 0.5               # Diffusion coefficient for ddx
d2 = 0.5               # Diffusion coefficient for ddy
A = (1/3)*spdiagm(0 => ones(size(Dxx, 1))) - dtn*(d1^2)*Dxx
B = (1/3)*spdiagm(0 => ones(size(Dyy, 1))) - dtn*(d2^2)*Dyy

# A = Matrix(A)
# B = Matrix(B)

# Create the initial data
U = @. 0.5 * exp(-400 * (X - 0.3)^2 - 400 * (Y - 0.35)^2 ) + 
     0.8 * exp(-400 * (X - 0.65)^2 - 400 * (Y - 0.5)^2 )

# Apply the SVD to the initial data
Vx_n, S_n, Vy_n = svd(U)
S_n = Diagonal(S_n)

# The initial data is rank two, but we could use information from
# the singular values
Vx_n = Vx_n[:, 1:2]
Vy_n = Vy_n[:, 1:2]
S_n = S_n[1:2,1:2]

# Call the Sylvester solver
max_iter = 10
max_rank = 100
max_size = 50

@btime begin

    Vx_nn, Vy_nn, S_nn, iter = extended_krylov_step(Vx_n, Vy_n, S_n, A, B, rel_eps, max_iter, max_rank, max_size)
    
end


# # Use this to check for a type instability
# @code_warntype extended_krylov_step(Vx_n, Vy_n, S_n, A, B, rel_eps, max_iter, max_rank)


# # Reset defaults for the number of samples and total time for
# # the benchmarking process
# BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
# BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

# benchmark_data = @benchmark extended_krylov_step(Vx_n, Vy_n, S_n, A, B, rel_eps, max_iter, max_rank)

# # Times are in nano-seconds (ns) which are converted to seconds
# sample_times = benchmark_data.times
# sample_times /= 10^9

# @printf "CPU results:\n"
# @printf "Minimum (s): %.8e\n" minimum(sample_times)
# @printf "Maximum (s): %.8e\n" maximum(sample_times)
# @printf "Median (s): %.8e\n" median(sample_times)
# @printf "Mean (s): %.8e\n" mean(sample_times)
# @printf "Standard deviation (s): %.8e\n" std(sample_times)


# # Profiling (code is assumed to already be compiled)
# # Initialize the profiler with a smaller sampling interval
# Profile.init()  # delay in seconds

# ProfileView.@profview begin
#     for iter = 1:50
#         extended_krylov_step(Vx_n, Vy_n, S_n, A, B, rel_eps, max_iter, max_rank)
#     end
# end 

# # open("./profile_data.txt", "w") do s
# #     Profile.print(C = false, IOContext(s, :displaysize => (24, 500)))
# # end
