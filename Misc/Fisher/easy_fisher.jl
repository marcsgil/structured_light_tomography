using LinearAlgebra, QuantumMeasurements, CairoMakie, Optim
includet("../../Utils/basis.jl")

N = 128
x = LinRange(-6, 6, N)
rs = Iterators.product(x, x)
pars = ((x[2] - x[1]), 0.0, 0.0, 1.0)

μ = assemble_measurement_matrix(Iterators.map(r -> positive_l_basis(4, r, pars), rs))
##
θs = zeros(size(μ, 2) - 1)
dest = similar(θs)

function probability(μ, θ)
    d = √(size(μ, 2))
    get_traceless_part(μ) * θ + get_trace_part(μ) / √d
end

function fisher_eigvals(μ, θ)
    inv_ps = probability(μ, θ)
    broadcast!(inv, inv_ps, inv_ps)
    D = Diagonal(inv_ps)
    T = get_traceless_part(μ)
    eigvals(T' * D * T)
end

opt = optimize(θ -> -sum(inv, fisher_eigvals(μ, θ)), θs)

opt.minimizer