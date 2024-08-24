using LinearAlgebra, BayesianTomography

function hermitian_transform!(ρ, L)
    rmul!(ρ, L)
    lmul!(L', ρ)
end

function density_matrix_transform!(ρ, L)
    hermitian_transform!(ρ, L)
    ρ ./= tr(ρ)
end

function transform_incomplete_povm!(incomplete_povm)
    g = sum(incomplete_povm)
    L = cholesky(g).L
    inv_L = inv(L)
    for Π ∈ incomplete_povm
        hermitian_transform!(Π, inv_L')
    end

    L
end

function η_func(θ, L)
    T = eltype(θ)
    dim = size(L, 1)
    ρ = density_matrix_reconstruction(θ)
    density_matrix_transform!(ρ, L)
    [real(tr(ρ * ω)) for ω ∈ GellMannMatrices(dim, complex(T))]
end

function η_func_jac(θ, L)
    T = eltype(θ)
    dim = size(L, 1)
    ρ = density_matrix_reconstruction(θ)
    hermitian_transform!(ρ, L)
    N = tr(ρ)
    ωs = GellMannMatrices(dim, complex(T))
    Ωs = [hermitian_transform!(ω, L) for ω ∈ ωs]

    [real(tr(Ω * ω) / N - tr(ρ * ω) * tr(Ω) / N^2) for ω ∈ ωs, Ω ∈ Ωs]
end

function incomplete_fisher(problem, θs, L)
    ηs = η_func(θs, L)
    J = η_func_jac(θs, L)
    J' * fisher(problem, ηs) * J
end

function blade_obstruction(x, y, blade_pos)
    x < blade_pos
end

function iris_obstruction(x, y, x₀, y₀, radius)
    (x - x₀)^2 + (y - y₀)^2 < radius^2
end

function inverse_iris_obstruction(x, y, x₀, y₀, radius)
    !iris_obstruction(x, y, x₀, y₀, radius)
end

function get_obstructed_basis(basis, obstruction_func, args...; kwargs...)
    map(basis) do f
        (x, y) -> f(x, y) * obstruction_func(x, y, args...; kwargs...)
    end
end

function get_valid_indices(x, y, obstruction_func, args...; kwargs...)
    findall(r -> obstruction_func(r..., args...; kwargs...), collect(Iterators.product(x, y)))
end