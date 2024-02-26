using LinearAlgebra, StaticArrays, KrylovKit

is_positive(A) = eigmin(A) ≥ 0
function ispossemidef!(A; tol=nextfloat(zero(real(eltype(A)))))
    for n ∈ 1:size(A, 1)
        A[n, n] += tol
    end
    isposdef(A)
end


eps(Float64)
real(ComplexF64)
##

#A = hermitianpart(A' * A)
#eigmin(A)
##
A = hermitianpart(rand(ComplexF64, 6, 6))
eigsolve(A, 1, :SR)

@benchmark eigsolve($A, 1, :SR)

eigvals(hermitianpart(A * A))

@benchmark is_positive($A)



A = [1 0; 0 0] |> hermitianpart

is_positive(A)
A + 1e-20I
@benchmark ispossemidef!($A)
@benchmark isposdef($A)

v = [1.0, 0]



ispossemidef!(v * v')

zero(Float64)
nextfloat(0.0)

F = cholesky(hermitianpart(A * A); check=false)
Cholesky
F.info