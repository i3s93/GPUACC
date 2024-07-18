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
    U = similar(U_old, size(U_old,1), max_size)
    V = similar(V_old, size(V_old,1), max_size)
    U[:,1:U_ncols] .= U_old[:,1:U_ncols]
    V[:,1:V_ncols] .= V_old[:,1:U_ncols]

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
        # R is ignores from the output
        Q_U, _ = qr!(U_aug)
        Q_V, _ = qr!(V_aug)
        
        # To get the thin form, the compact structs for Q_U and Q_V
        # need to be cast as a matrix type
        Q_U = Matrix(Q_U)
        Q_V = Matrix(Q_V)

        # Get the current sizes of the bases
        U_ncols = size(Q_U,2)
        V_ncols = size(Q_V,2)

        U[:,1:U_ncols] .= Q_U
        V[:,1:V_ncols] .= Q_V

        # Build and solve the reduced system using the Sylvester solver
        A1U[:,1:U_ncols] .= A1*U[:,1:U_ncols]
        A2V[:,1:V_ncols] .= A2*V[:,1:V_ncols]

        A1_tilde[1:U_ncols,1:U_ncols] .= U[:,1:U_ncols]'*A1U[:,1:U_ncols]
        A2_tilde[1:V_ncols,1:V_ncols] .= V[:,1:V_ncols]'*A2V[:,1:V_ncols]
        B1_tilde[1:U_ncols,1:V_ncols] .= (U[:,1:U_ncols]'*U_old[:,:])*S_old[:,:]*(V_old[:,:]'*V[:,1:V_ncols])
        
        A1_local = A1_tilde[1:U_ncols,1:U_ncols]
        A2_local = A2_tilde[1:V_ncols,1:V_ncols]
        B1_local = B1_tilde[1:U_ncols,1:V_ncols]

        S1_local = sylvc(A1_local, A2_local, B1_local)::typeof(S1)

        S1[1:U_ncols,1:V_ncols] = S1_local

        # Check convergence of the solver using the spectral norm of the residual
        # RU*[-B1_tilde S1; S1 zeros(size(S1, 1), size(S1, 2))]*RV'
        # This requires the upper triangular matrices from the QR factorization here
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