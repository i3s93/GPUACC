include("Workspace.jl")

@static if @isdefined(CUDSS)
    const Solver_t = CudssSolver{T} where T
else
    const Solver_t = Nothing
end

mutable struct Coefficients{T1}
    # Coefficient matrices for the problem
    A1::T1
    A2::T1

    # Storage for the factorizations of the coefficient matrices
    FA1::Union{Factorization, Solver_t, Nothing}
    FA2::Union{Factorization, Solver_t, Nothing}

    # Custom constructor
    function Coefficients(A1::T1, A2::T1) where {T1}
        # Create an instance of Coefficients with FA1 and FA2 initialized to nothing
        new{T1}(A1, A2, nothing, nothing)
    end
end


