include("State.jl")
include("Workspace.jl")
include("SolverParameters.jl")

include("bases.jl")
include("sylvester.jl")

"""
Function that approximately solves the Sylvester equation
    A1 X + X A2' + B = 0
Using an extended Krylov subspace approach.

Input:
    state_old: SVD factors from the previous time level
    ws: A reusable workspace for the Krylov solver
    params: A parameter struct containing information about the solver (tol, rank, backend, etc.)

Output:
    state_new: SVD factors for the new time level

The iteration terminates early provided the following condition is satisfied:
    ||A1 X + X A2'- B|| < ||B|| * tol,
where the norms are applied in a spectral sense. The term on the RHS is computed 

Note: The residual is measured by projecting onto the low-dimensional subspaces to reduce the
complexity of its formation.
"""
@fastmath @views function extended_krylov_step!(state_old::State2D, ws::ExtendedKrylovWorkspace2D, params::SolverParameters)

    # Tolerance for the construction of the Krylov basis
    threshold = get_threshold(state_old.S, params.rel_tol, params.backend)

    # Precompute the LU factorizations of A1 and A2
    compute_LU_factorizations!(ws)

    # Storage for the Krylov bases
    initialize_bases!(ws, state_old)

    # Variables to track during construction of the Krylov subspaces
    converged = false
    num_iterations = 0

    for iter_count = 1:params.max_iter

        update_bases_and_orthogonalize!(ws, params.backend)

        residual_norm = build_and_solve_sylvester!(state_old, ws, params.backend)

        if residual_norm < threshold
            num_iterations = iter_count
            converged = true
            break
        end

        shuffle_iterates!(ws)

    end

    state_new = apply_svd_truncation!(ws, params)

    return state_new, num_iterations
end

"""
Computes the threshold for the termination of the iterative scheme. The threshold is defined
according to the spectral norm of a given state. Note that we absorb the term representing the
denominator in the tolerance. 
"""
function get_threshold(S::Matrix{T}, rel_tol::Real, backend::CPU_backend)::Real where {T <: AbstractFloat}
    return S[1,1]*rel_tol
end

@static if @isdefined(CUDA)
    function get_threshold(S::CuMatrix{T}, rel_tol::Real, backend::CUDA_backend) where {T <: AbstractFloat}
        return CUDA.@allowscalar S[1,1]*rel_tol
    end

    function get_threshold(S::CuMatrix{T}, rel_tol::Real, backend::CUDA_UVM_backend) where {T <: AbstractFloat}
        return CUDA.@allowscalar S[1,1]*rel_tol
    end
end
