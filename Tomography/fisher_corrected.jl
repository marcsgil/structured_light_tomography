using LinearAlgebra, BayesianTomography
includet("../Utils/basis.jl")

function fisher_density!(dest, basis, operator_basis, ρ, dA)
    fill!(dest, 0)
    for n ∈ axes(dest, 4), m ∈ axes(dest, 3)
        Ωm = view(operator_basis, :, :, m)
        Ωn = view(operator_basis, :, :, n)
        for y ∈ axes(basis, 2), x ∈ axes(basis, 1)
            v = view(basis, x, y, :)
            dest[x, y, m, n] += real(dot(v, Ωm, v) * dot(v, Ωn, v) / dot(v, ρ, v))
        end
    end
    dest .*= dA
    nothing
end

obstructed_indices(xs, ys, filter_func, args...; kwargs...) = filter(I -> filter_func(xs[I[1]], ys[I[2]], args...; kwargs...), CartesianIndices((length(xs), length(ys))))

function gram_matrix(basis, xs, ys, filter_func, args...; kwargs...)
    idxs = obstructed_indices(xs, ys, filter_func, args...; kwargs...)
    sum(v * v' for v ∈ eachslice(view(basis, idxs, :), dims=1)) .* (xs[2] - xs[1]) * (ys[2] - ys[1])
end

function blade_obstruction(x, y, blade_pos)
    x < blade_pos
end

function iris_obstruction(x, y, x₀, y₀, radius)
    (x - x₀)^2 + (y - y₀)^2 < radius^2
end
##
basis_func = positive_l_basis(2, [0, 0, 1.0f0, 1])
rs = LinRange(-4.0f0, 4, 256)
dA = (rs[2] - rs[1])^2
basis = [f(x, y) for x ∈ rs, y ∈ rs, f ∈ basis_func]

ρ = [0 0; 0 1]

operator_basis = gell_mann_matrices(2, include_identity=false)

dest = zeros(real(eltype(basis)), size(basis, 1), size(basis, 2), 3, 3)
fisher_density!(dest, basis, operator_basis, ρ, dA)
##
r = 0.1f0

g = gram_matrix(basis, rs, rs, iris_obstruction, 0, 0, r)
idxs = obstructed_indices(rs, rs, iris_obstruction, 0, 0, r)

F = zeros(real(eltype(basis)), 3, 3)

for n ∈ axes(F, 2), m ∈ axes(F, 1)
    F[m, n] = sum(view(dest, idxs, m, n))
end

correction = [real(g ⋅ Ω1 * g ⋅ Ω2) for Ω1 ∈ eachslice(operator_basis, dims=3), Ω2 ∈ eachslice(operator_basis, dims=3)]

N = real(g ⋅ ρ)

F = F / N + correction / N^2

tr(inv(F))