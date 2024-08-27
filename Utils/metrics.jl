using BayesianTomography, SpecialFunctions, LinearAlgebra
using FiniteDiff: finite_difference_gradient

function σ_multiplier(conf_int)
    erfinv(conf_int) * oftype(conf_int, √2)
end

function fidelity_metric(θ, θ_pred, cov, conf_int)
    ρ = density_matrix_reconstruction(θ)
    grad = finite_difference_gradient(x -> fidelity(x, ρ), θ_pred)
    fid = fidelity(ρ, θ_pred)
    fid, σ_multiplier(conf_int) * √dot(grad, cov, grad)
end

function square_hilbert_schmidt_metric(θ, θ_pred, cov, conf_int)
    d = mapreduce((x, y) -> abs2(x - y), +, θ_pred, θ)
    d, σ_multiplier(conf_int) * sqrt(2 * (cov ⋅ cov))
end

function hilbert_schmidt_metric(θ, θ_pred, cov, conf_int)
    d = sqrt(mapreduce((x, y) -> abs2(x - y), +, θ_pred, θ))
    grad = θ_pred - θ
    d, σ_multiplier(conf_int) * √dot(grad, cov, grad)
end