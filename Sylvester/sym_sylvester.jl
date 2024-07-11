"""
A solver for the Sylvester equation
    AX + XB = C,
where the coefficient matrices A and B are assumed
to be symmetric and may or may not be sparse.

The new problem to solve is
    Lambda_A*X_tilde + X_tilde*Lambda_B, 
where 
    X_tilde = V_A'*X*V_B,
with Lambda_A and Lambda_B being diagonal matrices.
"""
@fastmath @views function sym_sylvc(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)

    m, n = size(C);
    [m; n] == LinearAlgebra.checksquare(A, B) || throw(DimensionMismatch("A, B and C have incompatible dimensions"))

    # First compute the eigenvalue decompositions of A and B
    # They are guaranteed to be diagonalizable, since they are symmetric.
    # Note the use of the "Symmetric" hint. It prevents a type instability
    # lambda_A, V_A = eigen(Symmetric(A))
    # lambda_B, V_B = eigen(Symmetric(B))

    # lambda_A, V_A = eigen(Symmetric(A))
    # lambda_B, V_B = eigen(Symmetric(B))

    FA = schur(Symmetric(A))
    FB = schur(Symmetric(B))

    lambda_A = FA.values
    lambda_B = FB.values

    V_A = FA.Z
    V_B = FB.Z

    # Transform the problem into one which is diagonal, starting with the RHS
    B_tilde = V_A'*B*V_B

    # Solve for X_tilde using broadcasting operations
    X_tilde = B_tilde./(lambda_A .+ lambda_B')

    # Transform the solution to recover x
    X = V_A*X_tilde*V_B'

    return X

end

# @fastmath @views function sym_sylvc(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)

#     m, n = size(C);
#     [m; n] == LinearAlgebra.checksquare(A, B) || throw(DimensionMismatch("A, B and C have incompatible dimensions"))

#     # First compute the eigenvalue decompositions of A and B
#     # They are guaranteed to be diagonalizable, since they are symmetric.
#     A_eig = eigen(Symmetric(A))
#     B_eig = eigen(Symmetric(B))

#     lambda_A = A_eig.values
#     V_A = A_eig.vectors
#     lambda_B = B_eig.values
#     V_B = B_eig.vectors    

#     # Transform the problem into one which is diagonal, starting with the RHS
#     B_tilde = V_A' * B * V_B

#     # Solve for X_tilde using broadcasting operations
#     X_tilde = B_tilde ./ (lambda_A .+ lambda_B')

#     # Transform the solution to recover x
#     X = V_A * X_tilde * V_B'

#     return X

# end

