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
complexity of its formation.
"""
@fastmath @views function sylvester_extended_krylov(U_old::AbstractMatrix, V_old::AbstractMatrix, S_old::AbstractMatrix, 
                                             A1::AbstractMatrix, A2::AbstractMatrix, 
                                             rel_eps::Real, max_iter::Int, max_rank::Int)

    # Tolerance for the construction of the Krylov basis
    threshold = opnorm(S_old)*rel_eps

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

    # Declarations for scoping
    S1 = similar(S_old) 
    U = copy(U_old) # Initialize the Krylov bases for U
    V = copy(V_old) # Initialize the Krylov bases for V

    # Variables to track during construction of the Krylov subspaces
    converged = false
    num_iterations = 0

    for iter_count = 1:max_iter

        # Extended Krylov bases and concatenate with the existing bases
        mul!(A1U_curr, A1, A1U_prev)
        ldiv!(inv_A1U_curr, FA1, inv_A1U_prev)

        mul!(A2V_curr, A2, A2V_prev)
        ldiv!(inv_A2V_curr, FA2, inv_A2V_prev)

        U_ext = hcat(U, A1U_curr, inv_A1U_curr)
        V_ext = hcat(V, A2V_curr, inv_A2V_curr)

        # Orthogonalize the augmented bases
        U, _ = qr!(U_ext)
        V, _ = qr!(V_ext)

        # Convert the basis to matrices
        U = Matrix(U)
        V = Matrix(V)

        # Build and solve the reduced system using the Sylvester solver
        A1U = A1*U
        A2V = A2*V

        A1_tilde = U'*A1U
        A2_tilde = V'*A2V  
        B1_tilde = (U'*U_old)*S_old*(V_old'*V)
        
        S1 = sylvc(A1_tilde, A2_tilde, B1_tilde)

        # Check convergence of the solver
        _, RU = qr!(hcat(U, A1U))
        _, RV = qr!(hcat(V, A2V))

        residual = RU*[-B1_tilde S1; S1 zeros(size(S1, 1), size(S1, 2))]*RV'

        if opnorm(residual) < threshold
            num_iterations = iter_count
            converged = true
            break
        end

        copy!(A1U_prev, A1U_curr)
        copy!(inv_A1U_prev, inv_A1U_curr)

        copy!(A2V_prev, A2V_curr)
        copy!(inv_A2V_prev, inv_A2V_curr)

    end

    # Perform SVD truncation on the dense solution tensor
    U_tilde, S_tilde, V_tilde = svd!(S1)

    # Here S_tilde is a vector, so we do this before
    # we promote S_tilde to a diagonal matrix
    #r = findlast(x -> x > rel_eps, S_tilde./maximum(S_tilde))
    r = find_last_rank(S_tilde./maximum(S_tilde), rel_eps, max_rank)

    # Define the new core tensor
    S_new = diagm(S_tilde[1:r])

    # Join the orthogonal bases from the QR and SVD steps
    U_new = U*view(U_tilde, :, 1:r)
    V_new = V*view(V_tilde, :, 1:r)

    return U_new, V_new, S_new, num_iterations

end


"""
A helper function for performing SVD truncation. The Input
array is assumed to be sorted in descending order, following
the usual convention of the SVD.

It finds the last index of an array that exceeds the tolerance tol.
"""
@fastmath function find_last_rank(x::AbstractArray, tol::Real, max_rank::Int)
    
    # Initialize the index for the rank
    r = 1

    # Since the array is ordered, we can break on the first instance
    @inbounds for i in eachindex(x)
        if x[i] < tol
            r = i
            break
        end
    end

    # If the rank exceeds the maximum, use the max_rank instead
    r = min(r, max_rank)

    return r
 end









# """
# Approximately solve
#     A X + X B' + Vx_n*S_n*Vy_n' = 0

# Input:
#     A: coeff matrix,
#     B: coeff matrix,
#     Vx_n, S_n, and Vy_n: SVD factors of initial matrix data.
#     tol stopping tolerance, with stopping criterion
#           ||A X  + X B'-Vx_n*S_n*Vy_n'||
#           ----------------------------------------  < tol
#                    ||Vx_n*S_n*Vy_n'||
#     computed in a cheap manner.

# Output:
#     solution factor   X = Vx_nn*S_nn*Vy_nn'
# """
# @fastmath @views function sylvester_extended_krylov(Vx_n::AbstractMatrix, Vy_n::AbstractMatrix, S_n::AbstractMatrix, 
#                                                     A::AbstractMatrix, B::AbstractMatrix, tol::Real)

#     # Precompute the 2-norm of SVD from the previous level
#     normb = opnorm(S_n)

#     # Precompute the LU factorization of A and B
#     FA = lu(A)
#     FB = lu(B)

#     DxVxn = similar(Vx_n)
#     DyVyn = similar(Vy_n)

#     Dxinv = similar(Vx_n)
#     Dyinv = similar(Vy_n)

#     # Create these variables for storing output
#     S1 = similar(S_n)
#     Vx_nn = similar(Vx_n)
#     Vy_nn = similar(Vy_n)
#     S_nn = similar(S_n)

#     # Variables to track during construction of the Krylov subspaces
#     converged = false
#     iter_count = 0

#     # Tolerance for the construction of the Krylov basis
#     threshold = normb*tol

#     while !converged

#         # Construct the extended Krylov bases
#         DxVxn = A*DxVxn
#         DyVyn = B*DyVyn

#         Dxinv = FA\Dxinv
#         Dyinv = FB\Dyinv

#         # Enrich the basis used for the Krylov subspaces
#         if iter_count == 0
            
#             Vx_aug = hcat(Vx_n, DxVxn, Dxinv)
#             Vy_aug = hcat(Vy_n, DyVyn, Dyinv)

#         else
            
#             Vx_aug = hcat(Vx_nn, DxVxn, Dxinv)
#             Vy_aug = hcat(Vy_nn, DyVyn, Dyinv)

#         end

#         # Orthogonalize the augmented bases
#         Vx_nn, _ = qr(Vx_aug)
#         Vy_nn, _ = qr(Vy_aug)

#         # Build and solve the reduced system using the Sylvester solver
#         AV = A*Vx_nn
#         A11 = Vx_nn'*AV
#         BV = B*Vy_nn
#         B11 = Vy_nn'*BV    
#         RHS = (Vx_nn'*Vx_n)*S_n*(Vy_n'*Vy_nn)

#         S1 = sylvc(A11, B11, RHS)

#         # Check convergence of the solver
#         _, R11 = qr(hcat(Vx_nn, AV))
#         _, R22 = qr(hcat(Vy_nn, BV))

#         iter_count += 1

#         residual = R11 * [-RHS S1; S1 zeros(size(S1, 1), size(S1, 2))] * R22'
#         residual_norm = opnorm(residual)

#         if residual_norm < threshold
#             converged = true
#             break
#         end

#     end

#     t1, S1n, t2 = svd(S1)

#     # Here S1n is a vector so we do this before
#     # we promote S1n to a diagonal matrix
#     r = findlast(x -> x > tol, S1n./max(S1n))
#     S1n = Diagonal(S1n)

#     # Use views to prevent unnecessary copies
#     Vx_nn = Vx_nn*t1[:, 1:r]
#     Vy_nn = Vy_nn*t2[:, 1:r]
#     S_nn = S1n[1:r, 1:r]

#     return Vx_nn, Vy_nn, S_nn, iter_count

# end