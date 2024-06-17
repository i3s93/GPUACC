# This script tests a GPU matrix-matrix multiplication algorithm.
# It benchmarks the multiplication A*B, where A is an L x M matrix
# and B is an M x N matrix.

using ArgParse
using BenchmarkTools
using CUDA

# Retrieve the command line arguments
s = ArgParseSettings()

@add_arg_table s begin
    "-L", "--L"
        help = "Number of rows in A";
        arg_type = Int
        default = 256
    "-M", "--M"
        help = "Number of columns in A and rows of B";
        arg_type = Int
        default = 256
    "-N", "--N"
        help = "Number of columns of B";
        arg_type = Int
        default = 256
    "-s", "--s"
        help = "Number of samples to use for statistics"
        arg_type = Int
        default = 1000
end

# Parse the arguments and print them to the command line
parsed_args = parse_args(s)

println("Options used:")
for (arg,val) in parsed_args
    println("  $arg  =  $val")
end

# Get the individual command line arguments from the dictionary
L = parsed_args["L"]
M = parsed_args["M"]
N = parsed_args["N"]
s = parsed_args["s"]

# Set the number of samples
BenchmarkTools.DEFAULT_PARAMETERS.samples = s

# Setup the matrices for the operation
# We declare these as "cost" so they are not treated as globals
# Alternatively, we could have used interpolation here.
const A = CUDA.randn(Float64, (L, M))
const B = CUDA.randn(Float64, (M, N))

# We can just use the "times" operator
benchmark_data = @benchmark CUDA.@sync A*B

# Times are in nano-seconds (ns) which are converted to seconds
sample_times = benchmark_data.times
sample_times /= 10^9

println("\n")
println("Minimum (s): ", minimum(sample_times))
println("Maximum (s): ", maximum(sample_times))
println("Median (s): ", median(sample_times))
println("Mean (s): ", mean(sample_times))
println("Standard Deviation (s): ", std(sample_times))
