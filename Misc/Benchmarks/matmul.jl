using CUDA, LinearAlgebra, BenchmarkTools

##CPU L = 64
L = 64
d = 6

A = rand(Float32, d^2, 2 * L^2)
x = rand(Float32, 2 * L^2)
y = zeros(Float32, d^2)

@benchmark mul!($y, $A, $x)

##GPU L = 64
cA = CUDA.rand(Float32, d^2, 2 * L^2)
cx = CUDA.rand(Float32, 2 * L^2)
cy = CUDA.zeros(Float32, d^2)

@benchmark CUDA.@sync mul!($cy, $cA, $cx)
##CPU L = 400
L = 400
d = 6

A = rand(Float32, d^2, 2 * L^2)
x = rand(Float32, 2 * L^2)
y = Vector{Float32}(undef, d^2)

@benchmark mul!($y, $A, $x)
##GPU L = 400
cA = CUDA.rand(Float32, d^2, 2 * L^2)
cx = CUDA.rand(Float32, 2 * L^2)
cy = CUDA.zeros(Float32, d^2)

@benchmark CUDA.@sync mul!($cy, $cA, $cx)
##
A = rand(Float32, d^2, 2 * L^2)
pinv(A)
@b pinv($A)