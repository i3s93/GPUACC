include("State.jl")
include("Workspace.jl")
include("SolverParameters.jl")

include("bases.jl")
include("sylvester.jl")

# TO-DO: Add A1 and A2 as members of the workspace, as well as their factorizations.
# The additional arguments can be removed from this function call.
@fastmath @views function extended_krylov_step!(state_old::State2D, ws::ExtendedKrylovWorkspace2D, params::SolverParameters)

    # Unpack the previous low-rank state data
    U_old, S_old, V_old, r_old = state_old.U, state_old.S, state_old.V, state_old.r

    # Number of columns used in the updated bases (changes across iterations)
    U_ncols = r_old
    V_ncols = r_old

    # Tolerance for the construction of the Krylov basis
    threshold = S_old[1,1]*params.rel_tol

    # Precompute the LU factorizations of A1 and A2
    compute_LU_factorizations!(ws)

    # Storage for the Krylov bases
    initialize_bases!(ws, state_old)

    # Variables to track during construction of the Krylov subspaces
    converged = false
    num_iterations = 0

    for iter_count = 1:params.max_iter

        update_bases_and_orthogonalize!(ws)

        residual_norm = build_and_solve_sylvester!(state_old, ws, params.backend)

        if residual_norm < threshold
            num_iterations = iter_count
            converged = true
            break
        end

        shuffle_iterates!(ws)

    end

    # TO-DO: Isolate this part of the solver
    # Perform SVD truncation on the dense solution tensor
    U_tilde, S_tilde, V_tilde = svd!(ws.S1[1:ws.U_ncols,1:ws.V_ncols])

    # Here S_tilde is a vector, so we do this before
    # we promote S_tilde to a diagonal matrix
    # We can exploit the fact that S_tilde is ordered (descending)
    r_new = sum(S_tilde .> params.rel_tol*S_tilde[1])
    r_new = min(r_new, params.max_rank)

    # Define the new "core" tensor
    S_new = Diagonal(S_tilde[1:r_new])

    # Join the orthogonal bases from the QR and SVD steps
    U_new = ws.U[:,1:ws.U_ncols]*U_tilde[1:ws.U_ncols,1:r_new]
    V_new = ws.V[:,1:ws.V_ncols]*V_tilde[1:ws.V_ncols,1:r_new]

    # Create the updated low-rank state
    state_new = State2D(U_new, S_new, V_new, r_new)

    return state_new, num_iterations
end