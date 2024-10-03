using BayesianTomography, LinearAlgebra

includet("../Utils/basis.jl")
include("../Utils/position_operators.jl")

rs = Float32.(1:400)

for order ∈ 1:5
    basis_d = fixed_order_basis(order, (200, 200, 100))
    basis_c = fixed_order_basis(order, (200, 200, 100), -Float32(π) / 6)

    measurement = Measurement(assemble_position_operators(rs, rs, basis_d, basis_c))

    @show cond(measurement)
end