# """
# Approximately solve
#     A X + X B' + U_old*S_old*V_old' = 0

# Input:
#     A: coeff matrix,
#     B: coeff matrix,
#     U_old, S_old, and V_old: SVD factors of initial data.
#     tol stopping tolerance, with stopping criterion
#           ||A X  + X B' -U_old*S_old*V_old'||
#           ----------------------------------------  < tol
#                    ||U_old*S_old*V_old'||
#     computed in a cheap manner.

# Output:
#     solution factor   X = U_new*S_new*V_new'
# """
# function sylvester_extended_krylov(U_old::AbstractMatrix, V_old::AbstractMatrix, S_old::AbstractMatrix, 
#                                    A1::AbstractMatrix, A2::AbstractMatrix, 
#                                    tol::Real, max_iter::Int, max_rank::Int)

#     # Retrieve some sizes from the input data
#     U_extents = size(U_old)
#     V_extents = size(V_old)

#     # Precompute the norm of SVD from the previous level
#     @show normb = opnorm(S_old)

#     # Tolerance for the construction of the Krylov basis and flag for convergence
#     threshold = normb*tol
#     converged = false

#     # Precompute factorizations of the coefficient matrices
#     # FA1 = factorize(A1) # This results in an error with use of \ because no method is defined
#     # FA2 = factorize(A2) # This results in an error with use of \ because no method is defined
#     FA1 = lu(A1)
#     FA2 = lu(A2)

#     # We initialize a pair of matrices for each dimension, one corresponding
#     # to a matrix and another its inverse. The sizes of these matrices do not
#     # are identical to those of U_old and V_old, and will be reused during the extensions
#     A1U_prev = U_old
#     A2V_prev = V_old
#     A1U_curr = Matrix{Float64}(undef, U_extents[1], U_extents[2])
#     A2V_curr = Matrix{Float64}(undef, V_extents[1], V_extents[2])

#     A1U_inv_prev = U_old
#     A2V_inv_prev = V_old
#     A1U_inv_curr = Matrix{Float64}(undef, U_extents[1], U_extents[2])
#     A2V_inv_curr = Matrix{Float64}(undef, V_extents[1], V_extents[2])

#     # Preallocate matrices for the new bases and coefficients
#     # We don't know the size of these in advance
#     U_new = Matrix{Float64}(undef, U_extents[1], 2*max_rank)
#     V_new = Matrix{Float64}(undef, V_extents[1], 2*max_rank)
#     S_new = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)

#     # These counters need to be accessible in the outer scope
#     U_new_ncols = 0
#     V_new_ncols = 0

#     # Preallocate some space for the extension process. These can be
#     # reused to concatenate matrices that have the same number of rows 
#     U_ext = Matrix{Float64}(undef, U_extents[1], 2*max_rank)
#     V_ext = Matrix{Float64}(undef, V_extents[1], 2*max_rank)

#     # Copy the old bases to the extended arrays
#     # We also track offsets that change from iteration to iteration
#     # for the column slicing operations
#     U_ext[:,1:U_extents[2]] = U_old[:,1:U_extents[2]]
#     V_ext[:,1:V_extents[2]] = V_old[:,1:V_extents[2]]
    
#     column_offset = U_extents[2] # Same as V_extents[2]

#     # Storage for the residual
#     # This is not known in advance
#     residual = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)
#     residual_tmp = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)

#     # Products used in the residuals extension these are not iterated
#     # so there is no need to swap them for the next pass
#     A1U_new = Matrix{Float64}(undef, U_extents[1], 2*max_rank)
#     A2V_new = Matrix{Float64}(undef, V_extents[1], 2*max_rank)
    
#     # Variables used in the reduced Sylvester equation solver
#     # Also not known in advance 
#     A1_tilde = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)
#     A2_tilde = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)
#     B_tilde = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)

#     for iter_count = 1:2
#     #for iter_count = 1:1

#         @show iter_count

#         U_new_ncols, V_new_ncols = extend_bases!(U_ext, V_ext,
#                                                  U_new, V_new,
#                                                  A1U_curr, A2V_curr,
#                                                  A1U_inv_curr, A2V_inv_curr,
#                                                  A1U_prev, A2V_prev,
#                                                  A1U_inv_prev, A2V_inv_prev, 
#                                                  A1, A2, FA1, FA2,
#                                                  column_offset)

#         @printf "\nPrinting out the new Krylov bases\n"

#         @printf "\nBasis for U_new:\n"
#         display(U_new[:,1:U_new_ncols])

#         @printf "\nBasis for V_new:\n"
#         display(V_new[:,1:V_new_ncols])

#         # @printf "\nPreparing to solve the Sylvester equation\n"

#         projected_residual_norm = solve_reduced_sylvester!(residual, residual_tmp, S_new,
#                                                             U_ext, V_ext, 
#                                                             U_new, V_new, 
#                                                             U_old, V_old, S_old,
#                                                             A1_tilde, A2_tilde, B_tilde,
#                                                             A1U_new, A2V_new, A1, A2,
#                                                             U_new_ncols, V_new_ncols)

#         @printf "S_new\n"
#         display(S_new[1:U_new_ncols,1:V_new_ncols])

#         # @show projected_residual_norm
#         # @show threshold
#         @show projected_residual_norm  < threshold

#         # Check for convergence
#         if projected_residual_norm < threshold
#             converged = true
#             break
#         end

#         # Transfer the bases into the corresponding augmented arrays for the next step
#         # In the extension function, we don't need to do this copy!
#         U_ext[:,1:U_new_ncols] .= U_new[:,1:U_new_ncols]
#         V_ext[:,1:V_new_ncols] .= V_new[:,1:V_new_ncols]

#         # Reset the column offset for the next pass
#         column_offset = U_new_ncols

#         # Current iterates become previous in the next iteration
#         A1U_prev .= A1U_curr
#         A1U_inv_prev .= A1U_inv_curr
#         A2V_prev .= A2V_curr
#         A2V_inv_prev .= A2V_inv_curr

#     end

#     # SVD truncation step is performed in-place
#     U_S_new, sigma_S_new, V_S_new = svd!(S_new[1:U_new_ncols,1:V_new_ncols])

#     # Here sigma_S_new is a vector, so we do this before
#     # we promote sigma_S_new to a diagonal matrix
#     r_last = findlast(x -> x > tol, sigma_S_new./maximum(sigma_S_new))
#     S_new = diagm(sigma_S_new[1:r_last])

#     # Combine the matrices from the QR and SVD steps
#     U_new[:,1:r_last] = U_ext[:, 1:r_last]*U_S_new[1:r_last,1:r_last]
#     V_new[:,1:r_last] = V_ext[:, 1:r_last]*V_S_new[1:r_last,1:r_last]

#     if !converged
#         @printf "Reached maximum number of iterations without converging!\n"
#     end

#     return U_new[:,1:r_last], V_new[:,1:r_last], S_new[1:r_last,1:r_last]

#     return Nothing

# end

# """
# Calculates the bases K_m for the reduced Krylov solve.
# """
# function extend_bases!(U_ext, V_ext,
#                         U_new, V_new,
#                         A1U_curr, A2V_curr,
#                         A1U_inv_curr, A2V_inv_curr,
#                         A1U_prev, A2V_prev,
#                         A1U_inv_prev, A2V_inv_prev, 
#                         A1, A2, FA1, FA2,
#                         column_offset)

#     # Extend the Krylov subspaces in each dimension using in-place multiplication
#     A1U_curr = A1*A1U_prev
#     A1U_inv_curr = FA1\A1U_inv_prev

#     A2V_curr = A2*A2V_prev
#     A2V_inv_curr = FA2\A2V_inv_prev

#     # Transfer each extension to the corresponding augmented array
#     # using offsets (with increments) to slice into the appropriate columns
#     # This requires us to track the number of new items added to the arrays
    
#     # The most recent basis for U and V is already included in U_ext and V_ext
#     # So both initially have "column_offset" number of column vectors
#     U_ext_ncols = column_offset
#     V_ext_ncols = column_offset 

#     # Extensions for the U basis: first with A1, then inv(A1)
#     offset = column_offset + 1
#     ncols_new = size(A1U_curr, 2)
#     U_ext[:,offset:(offset + ncols_new - 1)] = A1U_curr[:,1:ncols_new]
#     offset += ncols_new
#     U_ext_ncols += ncols_new

#     ncols_new = size(A1U_inv_curr, 2)
#     U_ext[:,offset:(offset + ncols_new - 1)] = A1U_inv_curr[:,1:ncols_new]
#     U_ext_ncols += ncols_new
    
#     # Extensions for the V basis: first with A2, then inv(A2)
#     offset = column_offset + 1
#     ncols_new = size(A2V_curr, 2)
#     V_ext[:,offset:(offset + ncols_new - 1)] = A2V_curr[:,1:ncols_new]
#     offset += ncols_new
#     V_ext_ncols += ncols_new

#     ncols_new = size(A2V_inv_curr, 2)
#     V_ext[:,offset:(offset + ncols_new - 1)] = A2V_inv_curr[:,1:ncols_new]
#     V_ext_ncols += ncols_new

#     # Orthogonalize the augmented bases using the in-place QR decomposition
#     # This will overwrite the columns of U_ext and V_ext to save some memory 
#     FU = qr!(U_ext[:,1:U_ext_ncols])
#     FV = qr!(V_ext[:,1:V_ext_ncols])

#     # Extract the orthogonal basis
#     Q_U = Matrix(FU.Q)
#     Q_V = Matrix(FV.Q)

#     U_new_ncols = size(Q_U, 2)
#     V_new_ncols = size(Q_V, 2)

#     # We copy the basis from the QR step into U_new and V_new
#     # Now it will be safe to work with U_ext and V_ext again
#     U_new[:,1:U_new_ncols] = Q_U[:,1:U_new_ncols]
#     V_new[:,1:V_new_ncols] = Q_V[:,1:V_new_ncols]

#     return U_new_ncols, V_new_ncols
# end

# """
# Solves a reduced Sylvester equation by projecting on to the Krylov subspaces 
# """
# function solve_reduced_sylvester!(residual, residual_tmp, S_new,
#                                     U_ext, V_ext, 
#                                     U_new, V_new, 
#                                     U_old, V_old, S_old,
#                                     A1_tilde, A2_tilde, B_tilde,
#                                     A1U_new, A2V_new, A1, A2,
#                                     U_new_ncols, V_new_ncols)

#     A1U_new[:,1:U_new_ncols] = A1*U_new[:,1:U_new_ncols]
#     A2V_new[:,1:V_new_ncols] = A2*V_new[:,1:V_new_ncols]

#     display(A1U_new[:,1:U_new_ncols])
#     display(A2V_new[:,1:V_new_ncols])


#     # Append A1U and A2V to the corresponding augmented arrays
#     # using offsets (with increments) to slice into the appropriate columns
#     # These arrays will be used in the evaluation of the residual

#     # Join U_new and A1U_new
#     ncols_new = U_new_ncols
#     U_ext[:, 1:ncols_new] = U_new[:,1:ncols_new]
#     offset = ncols_new + 1
#     U_ext_ncols = ncols_new

#     ncols_new = U_new_ncols
#     U_ext[:,offset:(offset + ncols_new - 1)] = A1U_new[:,1:ncols_new]
#     U_ext_ncols += ncols_new

#     # Join V_new and A2V_new
#     ncols_new = V_new_ncols
#     V_ext[:,1:ncols_new] = V_new[:,1:ncols_new]
#     offset = ncols_new + 1
#     V_ext_ncols = ncols_new

#     ncols_new = V_new_ncols
#     V_ext[:,offset:(offset + ncols_new - 1)] = A2V_new[:,1:ncols_new]
#     V_ext_ncols += ncols_new

#     # Build the components of the reduced Sylvester equation
#     # A1_tilde*S_new + S_new*A2_tilde' = B
#     A1_tilde[1:U_new_ncols, 1:U_new_ncols] = U_new[:,1:U_new_ncols]'*A1U_new[:,1:U_new_ncols]
#     A2_tilde[1:V_new_ncols, 1:V_new_ncols] = V_new[:,1:V_new_ncols]'*A2V_new[:,1:V_new_ncols]
#     B_tilde[1:U_new_ncols, 1:V_new_ncols] = (U_new[:,1:U_new_ncols]'*U_old)*S_old*(V_old'*V_new[:,1:V_new_ncols])

#     display(U_new[:,1:U_new_ncols]'*A1U_new[:,1:U_new_ncols])
#     display(V_new[:,1:V_new_ncols]'*A2V_new[:,1:V_new_ncols])

#     # Solve for S_new
#     S_new[1:U_new_ncols,1:V_new_ncols] = sylvc(A1_tilde[1:U_new_ncols, 1:U_new_ncols], 
#                                                A2_tilde[1:V_new_ncols, 1:V_new_ncols], 
#                                                B_tilde[1:U_new_ncols, 1:V_new_ncols])


#     S = sylvc(A1_tilde[1:U_new_ncols, 1:U_new_ncols], 
#     A2_tilde[1:V_new_ncols, 1:V_new_ncols], 
#     B_tilde[1:U_new_ncols, 1:V_new_ncols])

#     #display(S)

#     # Compute the in-place QR factorizations for the extended bases
#     # Here we exploit the fact that the in-place QR overwrites the input
#     # so that the upper triangular factor is written in the output
#     _, RU = qr!(U_ext[:,1:U_ext_ncols])
#     _, RV = qr!(V_ext[:,1:V_ext_ncols])

#     # Form the residual of the system in blocks
#     B_tilde_extents = size(B_tilde[1:U_new_ncols, 1:V_new_ncols])
#     S_new_extents = size(S_new[1:U_new_ncols,1:V_new_ncols])

#     # Create views to the blocks to avoid copies
#     block11 = view(residual, 1:B_tilde_extents[1], 1:B_tilde_extents[2])
#     block12 = view(residual, 1:B_tilde_extents[1], (B_tilde_extents[2]+1):(B_tilde_extents[2]+S_new_extents[2]))
#     block21 = view(residual, (B_tilde_extents[1]+1):(B_tilde_extents[1]+S_new_extents[1]), 1:S_new_extents[2])
#     block22 = view(residual, (B_tilde_extents[1]+1):(B_tilde_extents[1]+S_new_extents[1]), 
#                    (B_tilde_extents[2]+1):(B_tilde_extents[2]+S_new_extents[2]))

#     # Copy data to the blocks using broadcasting operations
#     block11 .= -B_tilde[1:U_new_ncols, 1:V_new_ncols]
#     block12 .= S_new[1:U_new_ncols,1:V_new_ncols]
#     block21 .= S_new[1:U_new_ncols,1:V_new_ncols]
#     block22 .= 0.0

#     # Dimensions of the matrices for the in-place multiplications
#     residual_nrows = B_tilde_extents[1] + S_new_extents[1]
#     residual_ncols = B_tilde_extents[2] + S_new_extents[2]

#     # Compute the residual using in-place multiplications
#     residual_tmp_subview = view(residual_tmp, 1:size(RU, 1), 1:residual_ncols)
#     residual_subview = view(residual, 1:residual_nrows, 1:residual_ncols)

#     # Compute RU*block_matrix*RV' from left to right using a temporary
#     residual_tmp_subview = RU*residual_subview
#     residual_subview = residual_tmp_subview*RV'

#     projected_residual_norm = opnorm(residual_subview)

#     return projected_residual_norm
# end


# Original code.... 
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


