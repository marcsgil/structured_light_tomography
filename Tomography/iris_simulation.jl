using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/obstructed_measurements.jl")
includet("../Utils/model_fitting.jl")

ρs = sample(ProductMeasure(2), 100)

rs = LinRange(-3.0f0, 3.0f0, 256)

basis_func = positive_l_basis(2, [0.0f0, 0, 1, 1])
obstructed_basis = [(x, y) -> f(x, y) * iris_obstruction(x, y, 0, 0, 0.7f0) for f in basis_func]

T0, Ω0, L0 = assemble_povm_matrix(rs, rs, basis_func)
T, Ω, L = assemble_povm_matrix(rs, rs, obstructed_basis)

mthd0 = LinearInversion(T, Ω)
mthd = LinearInversion(T, Ω)

θ = Float32(1 / √2) * [0, 0, 0]
η = η_func(θ, Ω0, L, Ω0)


probs = get_probs(mthd, η)

ρ_pred, η_pred, _ = prediction(probs, mthd)

η_pred