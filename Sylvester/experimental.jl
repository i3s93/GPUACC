"""
Approximately solve
    A X + X B' + U_old*S_old*V_old' = 0

Input:
    A: coeff matrix,
    B: coeff matrix,
    U_old, S_old, and V_old: SVD factors of initial data.
    tol stopping tolerance, with stopping criterion
          ||A X  + X B' -U_old*S_old*V_old'||
          ----------------------------------------  < tol
                   ||U_old*S_old*V_old'||
    computed in a cheap manner.

Output:
    solution factor   X = U_new*S_new*V_new'
"""
function sylvester_extended_krylov(U_old::AbstractMatrix, V_old::AbstractMatrix, S_old::AbstractMatrix, 
                                   A1::AbstractMatrix, A2::AbstractMatrix, 
                                   tol::Real, max_iter::Int, max_rank::Int)

    # Retrieve some sizes from the input data
    U_extents = size(U_old)
    V_extents = size(V_old)

    # Precompute the norm of SVD from the previous level
    @show normb = opnorm(S_old)

    # Tolerance for the construction of the Krylov basis and flag for convergence
    threshold = normb*tol
    converged = false

    # Precompute factorizations of the coefficient matrices
    # FA1 = factorize(A1) # This results in an error with use of \ because no method is defined
    # FA2 = factorize(A2) # This results in an error with use of \ because no method is defined
    FA1 = lu(A1)
    FA2 = lu(A2)

    # We initialize a pair of matrices for each dimension, one corresponding
    # to a matrix and another its inverse. The sizes of these matrices do not
    # are identical to those of U_old and V_old, and will be reused during the extensions
    A1U_prev = U_old
    A2V_prev = V_old
    A1U_curr = Matrix{Float64}(undef, U_extents[1], U_extents[2])
    A2V_curr = Matrix{Float64}(undef, V_extents[1], V_extents[2])

    A1U_inv_prev = U_old
    A2V_inv_prev = V_old
    A1U_inv_curr = Matrix{Float64}(undef, U_extents[1], U_extents[2])
    A2V_inv_curr = Matrix{Float64}(undef, V_extents[1], V_extents[2])

    # Preallocate matrices for the new bases and coefficients
    # We don't know the size of these in advance
    U_new = Matrix{Float64}(undef, U_extents[1], 2*max_rank)
    V_new = Matrix{Float64}(undef, V_extents[1], 2*max_rank)
    S_new = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)

    # These counters need to be accessible in the outer scope
    U_new_ncols = 0
    V_new_ncols = 0

    # Preallocate some space for the extension process. These can be
    # reused to concatenate matrices that have the same number of rows 
    U_ext = Matrix{Float64}(undef, U_extents[1], 2*max_rank)
    V_ext = Matrix{Float64}(undef, V_extents[1], 2*max_rank)

    # Copy the old bases to the extended arrays
    # We also track offsets that change from iteration to iteration
    # for the column slicing operations
    U_ext[:,1:U_extents[2]] = U_old[:,1:U_extents[2]]
    V_ext[:,1:V_extents[2]] = V_old[:,1:V_extents[2]]
    
    column_offset = U_extents[2] # Same as V_extents[2]

    # Storage for the residual
    # This is not known in advance
    residual = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)
    residual_tmp = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)

    # Products used in the residuals extension these are not iterated
    # so there is no need to swap them for the next pass
    A1U_new = Matrix{Float64}(undef, U_extents[1], 2*max_rank)
    A2V_new = Matrix{Float64}(undef, V_extents[1], 2*max_rank)
    
    # Variables used in the reduced Sylvester equation solver
    # Also not known in advance 
    A1_tilde = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)
    A2_tilde = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)
    B_tilde = Matrix{Float64}(undef, 2*max_rank, 2*max_rank)

    for iter_count = 1:2
    #for iter_count = 1:1

        @show iter_count

        U_new_ncols, V_new_ncols = extend_bases!(U_ext, V_ext,
                                                 U_new, V_new,
                                                 A1U_curr, A2V_curr,
                                                 A1U_inv_curr, A2V_inv_curr,
                                                 A1U_prev, A2V_prev,
                                                 A1U_inv_prev, A2V_inv_prev, 
                                                 A1, A2, FA1, FA2,
                                                 column_offset)

        @printf "\nPrinting out the new Krylov bases\n"

        @printf "\nBasis for U_new:\n"
        display(U_new[:,1:U_new_ncols])

        @printf "\nBasis for V_new:\n"
        display(V_new[:,1:V_new_ncols])

        # @printf "\nPreparing to solve the Sylvester equation\n"

        projected_residual_norm = solve_reduced_sylvester!(residual, residual_tmp, S_new,
                                                            U_ext, V_ext, 
                                                            U_new, V_new, 
                                                            U_old, V_old, S_old,
                                                            A1_tilde, A2_tilde, B_tilde,
                                                            A1U_new, A2V_new, A1, A2,
                                                            U_new_ncols, V_new_ncols)

        @printf "S_new\n"
        display(S_new[1:U_new_ncols,1:V_new_ncols])

        # @show projected_residual_norm
        # @show threshold
        @show projected_residual_norm  < threshold

        # Check for convergence
        if projected_residual_norm < threshold
            converged = true
            break
        end

        # Transfer the bases into the corresponding augmented arrays for the next step
        # In the extension function, we don't need to do this copy!
        U_ext[:,1:U_new_ncols] .= U_new[:,1:U_new_ncols]
        V_ext[:,1:V_new_ncols] .= V_new[:,1:V_new_ncols]

        # Reset the column offset for the next pass
        column_offset = U_new_ncols

        # Current iterates become previous in the next iteration
        A1U_prev .= A1U_curr
        A1U_inv_prev .= A1U_inv_curr
        A2V_prev .= A2V_curr
        A2V_inv_prev .= A2V_inv_curr

    end

    # SVD truncation step is performed in-place
    U_S_new, sigma_S_new, V_S_new = svd!(S_new[1:U_new_ncols,1:V_new_ncols])

    # Here sigma_S_new is a vector, so we do this before
    # we promote sigma_S_new to a diagonal matrix
    r_last = findlast(x -> x > tol, sigma_S_new./maximum(sigma_S_new))
    S_new = diagm(sigma_S_new[1:r_last])

    # Combine the matrices from the QR and SVD steps
    U_new[:,1:r_last] = U_ext[:, 1:r_last]*U_S_new[1:r_last,1:r_last]
    V_new[:,1:r_last] = V_ext[:, 1:r_last]*V_S_new[1:r_last,1:r_last]

    if !converged
        @printf "Reached maximum number of iterations without converging!\n"
    end

    return U_new[:,1:r_last], V_new[:,1:r_last], S_new[1:r_last,1:r_last]

    return Nothing

end

"""
Calculates the bases K_m for the reduced Krylov solve.
"""
function extend_bases!(U_ext, V_ext,
                        U_new, V_new,
                        A1U_curr, A2V_curr,
                        A1U_inv_curr, A2V_inv_curr,
                        A1U_prev, A2V_prev,
                        A1U_inv_prev, A2V_inv_prev, 
                        A1, A2, FA1, FA2,
                        column_offset)

    # Extend the Krylov subspaces in each dimension using in-place multiplication
    A1U_curr = A1*A1U_prev
    A1U_inv_curr = FA1\A1U_inv_prev

    A2V_curr = A2*A2V_prev
    A2V_inv_curr = FA2\A2V_inv_prev

    # Transfer each extension to the corresponding augmented array
    # using offsets (with increments) to slice into the appropriate columns
    # This requires us to track the number of new items added to the arrays
    
    # The most recent basis for U and V is already included in U_ext and V_ext
    # So both initially have "column_offset" number of column vectors
    U_ext_ncols = column_offset
    V_ext_ncols = column_offset 

    # Extensions for the U basis: first with A1, then inv(A1)
    offset = column_offset + 1
    ncols_new = size(A1U_curr, 2)
    U_ext[:,offset:(offset + ncols_new - 1)] = A1U_curr[:,1:ncols_new]
    offset += ncols_new
    U_ext_ncols += ncols_new

    ncols_new = size(A1U_inv_curr, 2)
    U_ext[:,offset:(offset + ncols_new - 1)] = A1U_inv_curr[:,1:ncols_new]
    U_ext_ncols += ncols_new
    
    # Extensions for the V basis: first with A2, then inv(A2)
    offset = column_offset + 1
    ncols_new = size(A2V_curr, 2)
    V_ext[:,offset:(offset + ncols_new - 1)] = A2V_curr[:,1:ncols_new]
    offset += ncols_new
    V_ext_ncols += ncols_new

    ncols_new = size(A2V_inv_curr, 2)
    V_ext[:,offset:(offset + ncols_new - 1)] = A2V_inv_curr[:,1:ncols_new]
    V_ext_ncols += ncols_new

    # Orthogonalize the augmented bases using the in-place QR decomposition
    # This will overwrite the columns of U_ext and V_ext to save some memory 
    FU = qr!(U_ext[:,1:U_ext_ncols])
    FV = qr!(V_ext[:,1:V_ext_ncols])

    # Extract the orthogonal basis
    Q_U = Matrix(FU.Q)
    Q_V = Matrix(FV.Q)

    U_new_ncols = size(Q_U, 2)
    V_new_ncols = size(Q_V, 2)

    # We copy the basis from the QR step into U_new and V_new
    # Now it will be safe to work with U_ext and V_ext again
    U_new[:,1:U_new_ncols] = Q_U[:,1:U_new_ncols]
    V_new[:,1:V_new_ncols] = Q_V[:,1:V_new_ncols]

    return U_new_ncols, V_new_ncols
end

"""
Solves a reduced Sylvester equation by projecting on to the Krylov subspaces 
"""
function solve_reduced_sylvester!(residual, residual_tmp, S_new,
                                    U_ext, V_ext, 
                                    U_new, V_new, 
                                    U_old, V_old, S_old,
                                    A1_tilde, A2_tilde, B_tilde,
                                    A1U_new, A2V_new, A1, A2,
                                    U_new_ncols, V_new_ncols)

    A1U_new[:,1:U_new_ncols] = A1*U_new[:,1:U_new_ncols]
    A2V_new[:,1:V_new_ncols] = A2*V_new[:,1:V_new_ncols]

    display(A1U_new[:,1:U_new_ncols])
    display(A2V_new[:,1:V_new_ncols])


    # Append A1U and A2V to the corresponding augmented arrays
    # using offsets (with increments) to slice into the appropriate columns
    # These arrays will be used in the evaluation of the residual

    # Join U_new and A1U_new
    ncols_new = U_new_ncols
    U_ext[:, 1:ncols_new] = U_new[:,1:ncols_new]
    offset = ncols_new + 1
    U_ext_ncols = ncols_new

    ncols_new = U_new_ncols
    U_ext[:,offset:(offset + ncols_new - 1)] = A1U_new[:,1:ncols_new]
    U_ext_ncols += ncols_new

    # Join V_new and A2V_new
    ncols_new = V_new_ncols
    V_ext[:,1:ncols_new] = V_new[:,1:ncols_new]
    offset = ncols_new + 1
    V_ext_ncols = ncols_new

    ncols_new = V_new_ncols
    V_ext[:,offset:(offset + ncols_new - 1)] = A2V_new[:,1:ncols_new]
    V_ext_ncols += ncols_new

    # Build the components of the reduced Sylvester equation
    # A1_tilde*S_new + S_new*A2_tilde' = B
    A1_tilde[1:U_new_ncols, 1:U_new_ncols] = U_new[:,1:U_new_ncols]'*A1U_new[:,1:U_new_ncols]
    A2_tilde[1:V_new_ncols, 1:V_new_ncols] = V_new[:,1:V_new_ncols]'*A2V_new[:,1:V_new_ncols]
    B_tilde[1:U_new_ncols, 1:V_new_ncols] = (U_new[:,1:U_new_ncols]'*U_old)*S_old*(V_old'*V_new[:,1:V_new_ncols])

    display(U_new[:,1:U_new_ncols]'*A1U_new[:,1:U_new_ncols])
    display(V_new[:,1:V_new_ncols]'*A2V_new[:,1:V_new_ncols])

    # Solve for S_new
    S_new[1:U_new_ncols,1:V_new_ncols] = sylvc(A1_tilde[1:U_new_ncols, 1:U_new_ncols], 
                                               A2_tilde[1:V_new_ncols, 1:V_new_ncols], 
                                               B_tilde[1:U_new_ncols, 1:V_new_ncols])


    S = sylvc(A1_tilde[1:U_new_ncols, 1:U_new_ncols], 
    A2_tilde[1:V_new_ncols, 1:V_new_ncols], 
    B_tilde[1:U_new_ncols, 1:V_new_ncols])

    #display(S)

    # Compute the in-place QR factorizations for the extended bases
    # Here we exploit the fact that the in-place QR overwrites the input
    # so that the upper triangular factor is written in the output
    _, RU = qr!(U_ext[:,1:U_ext_ncols])
    _, RV = qr!(V_ext[:,1:V_ext_ncols])

    # Form the residual of the system in blocks
    B_tilde_extents = size(B_tilde[1:U_new_ncols, 1:V_new_ncols])
    S_new_extents = size(S_new[1:U_new_ncols,1:V_new_ncols])

    # Create views to the blocks to avoid copies
    block11 = view(residual, 1:B_tilde_extents[1], 1:B_tilde_extents[2])
    block12 = view(residual, 1:B_tilde_extents[1], (B_tilde_extents[2]+1):(B_tilde_extents[2]+S_new_extents[2]))
    block21 = view(residual, (B_tilde_extents[1]+1):(B_tilde_extents[1]+S_new_extents[1]), 1:S_new_extents[2])
    block22 = view(residual, (B_tilde_extents[1]+1):(B_tilde_extents[1]+S_new_extents[1]), 
                   (B_tilde_extents[2]+1):(B_tilde_extents[2]+S_new_extents[2]))

    # Copy data to the blocks using broadcasting operations
    block11 .= -B_tilde[1:U_new_ncols, 1:V_new_ncols]
    block12 .= S_new[1:U_new_ncols,1:V_new_ncols]
    block21 .= S_new[1:U_new_ncols,1:V_new_ncols]
    block22 .= 0.0

    # Dimensions of the matrices for the in-place multiplications
    residual_nrows = B_tilde_extents[1] + S_new_extents[1]
    residual_ncols = B_tilde_extents[2] + S_new_extents[2]

    # Compute the residual using in-place multiplications
    residual_tmp_subview = view(residual_tmp, 1:size(RU, 1), 1:residual_ncols)
    residual_subview = view(residual, 1:residual_nrows, 1:residual_ncols)

    # Compute RU*block_matrix*RV' from left to right using a temporary
    residual_tmp_subview = RU*residual_subview
    residual_subview = residual_tmp_subview*RV'

    projected_residual_norm = opnorm(residual_subview)

    return projected_residual_norm
end




