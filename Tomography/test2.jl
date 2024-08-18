using StructuredLight, BayesianTomography, Tullio, LinearAlgebra, CairoMakie, ForwardDiff
includet("../Utils/basis.jl")

function blade_obstruction(x, y, blade_pos)
    x < blade_pos
end

function iris_obstruction(x, y, x₀, y₀, radius)
    (x - x₀)^2 + (y - y₀)^2 < radius^2
end




##
rs = LinRange(-3.0f0, 3, 512)
basis_func = [(x, y) -> f(x, y) * iris_obstruction(x, y, 0, 0, 0.3) for f in positive_l_basis(2, [0.0f0, 0, 1, 1])]
buffer = [f(rs[1], rs[1]) for f in basis_func]

T, Ω, L = assemble_povm_matrix(basis_func, rs, rs)
povm = LinearInversion(T, pinv(T), Ω)

ρ = [1.0f0 0; 0 0]
img = Matrix{Float32}(undef, length(rs), length(rs))

get_intensity!(img, buffer, ρ, basis_func, rs, rs)
normalize!(img, 1)

σ, ηs, cov = prediction(img, povm)

ρ = project2density(σ / tr(σ))


ω = gell_mann_matrices(2)
η_func([1 / √2, 1 / √2, 0, 0], ω, L, ω)
ηs

J = ForwardDiff.jacobian(θ -> η_func(θ, ω, L, ω), [1 / √2, 1 / √2, 0, 0])[2:end, 2:end]


η_func_jac([1 / √2, 1 / √2, 0, 0], ω, L, ω)[2:end, 2:end] ≈ J

@time η_func([1 / √2, 1 / √2, 0, 0], ω, L, ω)

η_func_jac([1 / √2, 1 / √2, 0, 0], ω, L, ω)
##
rs = LinRange(-4.0f0, 4, 256)
basis_func = [(x, y) -> f(x, y) * blade_obstruction(x, y, Inf) for f in positive_l_basis(2, [0.0f0, 0, 1, 1])]
img = Matrix{Float32}(undef, length(rs), length(rs))
buffer = [f(rs[1], rs[1]) for f in basis_func]
T, Ω, L = assemble_povm_matrix(basis_func, rs, rs)
ω = gell_mann_matrices(2)
F = Matrix{Float32}(undef, 3, 3)
coords = LinRange(-1.0f0, 1, 100)
inv_sqrt_2 = Float32(1 / √2)
θs = [[inv_sqrt_2, x * inv_sqrt_2, 0, z * inv_sqrt_2] for x in coords, z ∈ coords]



fisher_values = Matrix{Float32}(undef, length(coords), length(coords))

Threads.@threads for n ∈ eachindex(fisher_values)
    img = Matrix{Float32}(undef, length(rs), length(rs))
    buffer = Vector{ComplexF32}(undef, length(basis_func))
    F = Matrix{Float32}(undef, 3, 3)
    fisher_values[n] = fisher_at(θs[n], img, buffer, F, basis_func, Ω, L, ω, rs, T)
end
##
fig = Figure(size=(700, 600))
ax = Axis(fig[1, 1], aspect=DataAspect())
hm = heatmap!(ax, coords, coords, fisher_values)
Colorbar(fig[1, 2], hm)
fig
##
basis_func = [(x, y) -> f(x, y) * iris_obstruction(x, y, 0,0,0.1) for f in positive_l_basis(2, [0.0f0, 0, 1, 1])]
T, Ω, L = assemble_povm_matrix(basis_func, rs, rs)
ω = gell_mann_matrices(2)
coords = LinRange(-1.0f0, 1, 100)
inv_sqrt_2 = Float32(1 / √2)
θs = [[inv_sqrt_2, x * inv_sqrt_2, 0, z * inv_sqrt_2] for x in coords, z ∈ coords]


fisher_values_obs = Matrix{Float32}(undef, length(coords), length(coords))

Threads.@threads for n ∈ eachindex(fisher_values)
    img = Matrix{Float32}(undef, length(rs), length(rs))
    buffer = Vector{ComplexF32}(undef, length(basis_func))
    F = Matrix{Float32}(undef, 3, 3)
    fisher_values_obs[n] = fisher_at(θs[n], img, buffer, F, basis_func, Ω, L, ω, rs, T)
end
##
fig = Figure(size=(700, 600))
ax = Axis(fig[1, 1], aspect=DataAspect())
hm = heatmap!(ax, coords, coords, fisher_values_obs)
Colorbar(fig[1, 2], hm)
fig