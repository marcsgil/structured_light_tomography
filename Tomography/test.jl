using LinearAlgebra

A = rand(Float32, 2 * 400^2, 35)
b = rand(Float32, 2 * 400^2)
θ = Array{Float32}(undef, 35)

C = Array{Float32}(undef, 35, 35)
d = Array{Float32}(undef, 35)
##

f(A, b) = A \ b

function g!(C, d, A, b)
    mul!(C, A', A)
    mul!(d, A', b)
    C \ d
end

f(A, b) ≈ g!(C, d, A, b)

@benchmark f($A, $b)
@benchmark g!($C, $d, $A, $b)
##

