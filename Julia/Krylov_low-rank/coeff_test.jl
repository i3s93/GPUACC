using LinearAlgebra

include("Coefficients.jl")

# Create an instance of Coefficients with FA1 and FA2 initialized to nothing
A1 = randn(Float64,2,2)
A2 = randn(Float64,4,4)
coeff = Coefficients(A1, A2)

# Initially, FA1 and FA2 are nothing
println("coeff.FA1:")
display(coeff.FA1)
println("coeff.FA2:")
display(coeff.FA2)

# Later, assign factorizations to FA1 and FA2
coeff.FA1 = qr(A1)
coeff.FA2 = qr(A2)

println("coeff.FA1:")
display(coeff.FA1)
println("coeff.FA2:")
display(coeff.FA2)