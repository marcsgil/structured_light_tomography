using LinearAlgebra

A = rand(Float32, 10^4, 10^2)
invA = pinv(A)
b = rand(Float32, 10^4)

F = qr(A)

@benchmark pinv($A)
@benchmark qr($A, ColumnNorm())

dest1 = Vector{Float32}(undef, 10^2)
dest2 = Vector{Float32}(undef, 10^2)

ldiv!(dest1, F, b)
mul!(dest2, invA, b)

dest1 ≈ dest2 # true

@benchmark ldiv!($dest1, $F, $b)
@benchmark mul!($dest2, $invA, $b)