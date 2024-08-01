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
#using Profile
#using ProfileView

"""
Helper function to create the 2D grid
"""
function ndgrid(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Number}
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

"""
Approximately solve:
    A1 X + X A2' + U_old*S_old*V_old' = 0

Input:
    U_old, S_old, and V_old: SVD factors from the previous time level
    A1, A2: coefficient matrices
    rel_eps: Relative truncation tolerance for SVD truncation
    max_iter: Maximum number of Krylov iterations
    max_rank: Maximumm rank for SVD truncation

Output:
    U_new, S_new, V_new: Solution factors for the new time level

The iteration terminates early provided the following condition is satisfied:
    ||A1 X + X A2'- U_old*S_old*V_old'|| < ||U_old*S_old*V_old'|| * tol.
This quantity is measured by projecting onto the low-dimensional subspaces to reduce the
complexity of its formation. We use the spectral norm here.
"""
@fastmath @views function extended_krylov_step(U_old::AbstractMatrix, V_old::AbstractMatrix, S_old::AbstractMatrix, 
                                             A1::AbstractMatrix, A2::AbstractMatrix, 
                                             rel_eps::Real, max_iter::Int, max_rank::Int, max_size::Int = 256)

    # Number of columns used in the updated bases (changes across iterations)
    U_ncols = size(U_old,2)
    V_ncols = size(V_old,2)

    # Tolerance for the construction of the Krylov basis
    threshold = opnorm(S_old,2)*rel_eps

    # Precompute the LU factorizations of A1 and A2
    FA1 = lu(A1)
    FA2 = lu(A2)

    # Storage for the Krylov bases
    U = similar(U_old, size(U_old,1), max_size)
    V = similar(V_old, size(V_old,1), max_size)
    U[:,1:U_ncols] .= U_old[:,1:U_ncols]
    V[:,1:V_ncols] .= V_old[:,1:U_ncols]

    # Storage for A1^{k}U and A2^{k}V for two iterates
    A1U_prev = copy(U_old)
    A2V_prev = copy(V_old)
    A1U_curr = similar(U_old)
    A2V_curr = similar(V_old)

    # Storage for A1^{-k}U and A2^{-k}V for two iterates
    inv_A1U_prev = copy(U_old)
    inv_A2V_prev = copy(V_old)
    inv_A1U_curr = similar(U_old)
    inv_A2V_curr = similar(V_old)

    # Storage for the solution of the reduced Sylvester equation
    S1 = similar(S_old, max_size, max_size) 
    A1_tilde = similar(U_old, max_size, max_size)
    A2_tilde = similar(U_old, max_size, max_size)
    B1_tilde = similar(U_old, max_size, max_size)

    # Preallocate and initialize the data for the evaluation of the residual
    A1U = similar(U_old, size(U_old,1), max_size)
    A2V = similar(V_old, size(V_old,1), max_size)

    block_matrix = similar(U_old, max_size, max_size)
    residual = similar(U_old, max_size, max_size)

    # Variables to track during construction of the Krylov subspaces
    converged = false
    num_iterations = 0

    for iter_count = 1:max_iter

        # Extended Krylov bases and concatenate with the existing bases
        #
        # TO-DO: Define "batched" versions of these triangular solves for
        # the inverse operation so they can be performed concurrently
        mul!(A1U_curr, A1, A1U_prev)
        ldiv!(inv_A1U_curr, FA1, inv_A1U_prev)

        mul!(A2V_curr, A2, A2V_prev)
        ldiv!(inv_A2V_curr, FA2, inv_A2V_prev)

        U_aug = hcat(U[:,1:U_ncols], A1U_curr, inv_A1U_curr)
        V_aug = hcat(V[:,1:V_ncols], A2V_curr, inv_A2V_curr)

        # Orthogonalize the augmented bases
        # Note that we don't explicitly need to cast to the appropriate type
        # when we transfer the bases
        F_U = qr!(U_aug)
        F_V = qr!(V_aug)

        # Get the current sizes of the bases and transfer accordingly
        U_ncols = size(F_U,2)
        V_ncols = size(F_V,2)

        U[:,1:U_ncols] .= F_U.Q[:,1:U_ncols]
        V[:,1:V_ncols] .= F_V.Q[:,1:V_ncols]

        # Build and solve the reduced system using the Sylvester solver
        A1U[:,1:U_ncols] .= A1*U[:,1:U_ncols]
        A2V[:,1:V_ncols] .= A2*V[:,1:V_ncols]

        A1_tilde[1:U_ncols,1:U_ncols] .= U[:,1:U_ncols]'*A1U[:,1:U_ncols]
        A2_tilde[1:V_ncols,1:V_ncols] .= V[:,1:V_ncols]'*A2V[:,1:V_ncols]
        B1_tilde[1:U_ncols,1:V_ncols] .= (U[:,1:U_ncols]'*U_old[:,:])*S_old[:,:]*(V_old[:,:]'*V[:,1:V_ncols])        
        S1[1:U_ncols,1:V_ncols] .= sylvc(A1_tilde[1:U_ncols,1:U_ncols], A2_tilde[1:V_ncols,1:V_ncols], B1_tilde[1:U_ncols,1:V_ncols])

        # Check convergence of the solver using the spectral norm of the residual
        # RU*[-B1_tilde S1; S1 zeros(size(S1, 1), size(S1, 2))]*RV'
        _, RU = qr!(hcat(U[:,1:U_ncols], A1U[:,1:U_ncols]))
        _, RV = qr!(hcat(V[:,1:V_ncols], A2V[:,1:V_ncols]))

        # Build the blocks of the matrix [-B1_tilde S1; S1 zeros(size(S1, 1), size(S1, 2))]
        block_matrix[1:U_ncols,1:V_ncols] .= -B1_tilde[1:U_ncols,1:V_ncols]
        block_matrix[U_ncols .+ (1:U_ncols),1:V_ncols] .= S1[1:U_ncols,1:V_ncols]
        block_matrix[1:U_ncols,V_ncols .+ (1:V_ncols)] .= S1[1:U_ncols,1:V_ncols]
        block_matrix[U_ncols .+ (1:U_ncols),V_ncols .+ (1:V_ncols)] .= 0.0

        # Create a reference to the relevant block of the residual
        residual_block = residual[1:size(RU,1),1:size(RV,2)]
        residual_block .= RU[:,:]*block_matrix[1:(2*U_ncols),1:(2*V_ncols)]*RV[:,:]'

        if opnorm(residual_block,2) < threshold
            num_iterations = iter_count
            converged = true
            break
        end

        A1U_prev .= A1U_curr
        inv_A1U_prev .= inv_A1U_curr

        A2V_prev .= A2V_curr
        inv_A2V_prev .= inv_A2V_curr

    end

    # Perform SVD truncation on the dense solution tensor
    U_tilde, S_tilde, V_tilde = svd!(S1[1:U_ncols,1:V_ncols])

    # Here S_tilde is a vector, so we do this before
    # we promote S_tilde to a diagonal matrix
    # We can exploit the fact that S_tilde is ordered (descending)
    r = sum(S_tilde .> rel_eps*S_tilde[1])
    r = min(r, max_rank)

    # Define the new core tensor
    S_new = Diagonal(S_tilde[1:r])

    # Join the orthogonal bases from the QR and SVD steps
    U_new = U[:,1:U_ncols]*U_tilde[1:U_ncols,1:r]
    V_new = V[:,1:V_ncols]*V_tilde[1:V_ncols,1:r]

    return U_new, V_new, S_new, num_iterations

end

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
