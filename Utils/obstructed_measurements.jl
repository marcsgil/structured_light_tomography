linear_combinations(f, c) = (x...; kw...) -> map(c, f) do c, f
    c * f(x...; kw...)
end |> sum

function gram_matrix(basis_func, x, y)
    ΔA = (x[2] - x[1]) * (y[2] - y[1])

    map(Iterators.product(basis_func, basis_func)) do pair
        f = pair[1]
        g = pair[2]
        sum(conj(f(x, y)) * g(x, y) for x ∈ x, y ∈ y) * ΔA
    end |> Hermitian
end

function transform_state(U, ρ)
    new_ρ = U' * ρ * U
    new_ρ ./= tr(new_ρ)
    Hermitian(new_ρ)
end

function build_orthonormal_basis(basis_func, x, y)
    G = gram_matrix(basis_func, x, y)
    U = cholesky(G).U
    p = inv(U)

    U, [linear_combinations(basis_func, c) for c ∈ eachcol(p)]
end

function blade_obstruction(x, y, blade_pos)
    x < blade_pos
end

function iris_obstruction(x, y, x₀, y₀, radius)
    (x - x₀)^2 + (y - y₀)^2 < radius^2
end

function get_obstructed_basis(basis, obstruction_func, args...; kwargs...)
    map(basis) do f
        (x, y) -> f(x, y) * obstruction_func(x, y, args...; kwargs...)
    end
end

function get_proper_povm_and_states(x, y, ρs, basis)
    G = gram_matrix(basis, x, y)
    U = cholesky(G).U
    P = inv(U)

    orthonormal_basis = [linear_combinations(basis, c) for c ∈ eachcol(P)]

    povm = assemble_position_operators(x, y, orthonormal_basis)
    povm, stack(ρ -> transform_state(U, ρ), eachslice(ρs, dims=3))
end

function get_intensity(ρ, basis_func, x, y)
    map(Iterators.product(x, y)) do r
        v = [f(r...) for f ∈ basis_func]
        real(dot(v, ρ, v))
    end
end
##
using BayesianTomography, LinearAlgebra, StructuredLight, CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")

function test_obstructed_measurement()
    rs = Base.oneto(200)
    x₀ = length(rs) ÷ 2
    y₀ = length(rs) ÷ 2
    w = length(rs) ÷ 8

    basis = positive_l_basis(2, [x₀, y₀, w, 1])
    d = length(basis)
    ρs = sample(ProductMeasure(d), 1)

    obstructed_basis = get_obstructed_basis(basis, iris_obstruction, x₀, y₀, 0.5w)

    G = gram_matrix(obstructed_basis, rs, rs)
    U = cholesky(G).U
    P = inv(U)

    orthonormal_basis = [linear_combinations(obstructed_basis, c) for c ∈ eachcol(P)]

    new_ρs = stack(ρ -> transform_state(U, ρ), eachslice(ρs, dims=3))

    for (ρ, new_ρ) ∈ zip(eachslice(ρs, dims=3), eachslice(new_ρs, dims=3))
        I1 = get_intensity(new_ρ, orthonormal_basis, rs, rs)
        I2 = get_intensity(ρ, obstructed_basis, rs, rs)
        factor = real(tr(ρ * G))

        @assert I1 * factor ≈ I2
    end
end

test_obstructed_measurement()