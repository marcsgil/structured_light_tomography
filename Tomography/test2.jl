using StructuredLight, BayesianTomography, Tullio, LinearAlgebra, CairoMakie
includet("../Utils/basis.jl")

function get_intensity(ρ, basis_func, x, y)
    map(Iterators.product(x, y)) do r
        v = [f(r...) for f ∈ basis_func]
        real(dot(v, ρ, v))
    end
end

function gram_matrix(basis, dA)
    @tullio g[j, k] := basis[x, y, j] * conj(basis[x, y, k]) * dA
end

function assemble_povm_matrix(basis_functions, xs, ys)
    dim = length(basis_functions)
    dA = (xs[2] - xs[1]) * (ys[2] - ys[1])
    basis = stack(f(x, y) for x in xs, y in ys, f in basis_functions)
    ω = gell_mann_matrices(dim, include_identity=true)
    g = gram_matrix(basis, dA)
    L = cholesky(g).L
    inv_L = inv(L)
    Ω = stack(inv_L' * ω * inv_L for ω ∈ eachslice(ω, dims=3))
    @tullio T[x, y, n] := Ω[r, s, n] * basis[x, y, r] * conj(basis[x, y, s]) * dA |> real
    reshape(T, :, size(T, 3)), Ω
end

function blade_obstruction(x, y, blade_pos)
    x < blade_pos
end

function iris_obstruction(x, y, x₀, y₀, radius)
    (x - x₀)^2 + (y - y₀)^2 < radius^2
end
##
rs = LinRange(-3.0f0, 3, 512)
basis_func = [(x, y) -> f(x, y) * iris_obstruction(x, y, 0, 0, 0.3) for f in positive_l_basis(2, [0.0f0, 0, 1, 1])]

T, Ω = assemble_povm_matrix(basis_func, rs, rs)
povm = LinearInversion(T, pinv(T), Ω)


ρ = [1 im; -im 1] / 2.0f0

img = get_intensity(ρ, basis_func, rs, rs)
normalize!(img, 1)

σ, ηs, cov = prediction(img, povm)

ρ = project2density(σ / tr(σ))

[real(tr(ρ * ω)) for ω ∈ eachslice(gell_mann_matrices(2), dims=3)]
