using LinearAlgebra
includet("../Utils/basis.jl")

function assemble_position_operators(xs, ys, basis)
    T = complex(float(eltype(xs)))
    Π = Matrix{T}(undef, length(basis), length(basis))
    operators = Matrix{Matrix{T}}(undef, length(xs), length(ys))

    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]
    ΔA = Δx * Δy

    for (n, y) ∈ enumerate(ys), (m, x) ∈ enumerate(xs)
        for (k, ψ) ∈ enumerate(basis), j ∈ eachindex(basis)
            ϕ = basis[j]
            Π[j, k] = conj(ϕ(x, y)) * ψ(x, y) * ΔA
        end
        operators[m, n] = copy(Π)
    end

    return operators
end

function hermitian_transform!(ρ, L)
    rmul!(ρ, L)
    lmul!(L', ρ)
    nothing
end

function density_matrix_transform!(ρ, L)
    hermitian_transform!(ρ, L)
    ρ ./= tr(ρ)
    nothing
end

function get_transformation_functions!(partial_povm)
    g = sum(partial_povm)
    L = cholesky(g).L
    inv_L = inv(L)
    for Π ∈ partial_povm
        rmul!(Π, inv_L')
        lmul!(inv_L, Π)
    end

    """function f!(ρ)
        rmul!(ρ, inv_L)
        lmul!(inv_L', ρ)
        ρ ./= tr(ρ)
    end"""

    L
end
##
rs = LinRange(-1, 1, 128)
basis = positive_l_basis(2, [0, 0, 1, 1])

X = randn(ComplexF64, 2, 2)
ρ = X * X'
ρ ./= tr(ρ)
σ = copy(ρ)

operators_base = assemble_position_operators(rs, rs, basis)
operators = deepcopy(operators_base)
L = get_transformation_functions!(operators)

f!(σ, L)

ρ
σ

probs1 = [real(tr(Π * ρ)) for Π ∈ operators_base]
probs2 = [real(tr(Π * σ)) for Π ∈ operators]

sum(probs1)
sum(probs2)


sum(operators)