using Tullio

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

    """basis = [ϕ(x, y) for x ∈ xs, y ∈ ys, ϕ ∈ basis]

    operators = map(eachslice(basis, dims=(1, 2))) do v
        conj.((v * v'))
    end"""

    return operators
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
    reshape(T, :, size(T, 3)), Ω, L
end

function get_intensity!(img, buffer, ρ, basis_func, x, y)
    for (n, y) ∈ enumerate(y), (m, x) ∈ enumerate(x)
        for (i, f) ∈ enumerate(basis_func)
            buffer[i] = f(x, y)
        end
        img[m, n] = real(dot(buffer, ρ, buffer))
    end
end

function η_func(θ, Ω, L, ω)
    transformed_ρ = L' * linear_combination(θ, Ω) * L
    N = tr(transformed_ρ)
    [real(tr(transformed_ρ * ω) / N) for ω ∈ eachslice(ω, dims=3)]
end

function η_func_jac(θ, Ω, L, ω)
    transformed_ρ = L' * linear_combination(θ, Ω) * L
    N = tr(transformed_ρ)

    [real(tr(L' * Ω * L * ω) / N - tr(transformed_ρ * ω) * tr(L' * Ω * L) / N^2)
     for ω ∈ eachslice(ω, dims=3), Ω ∈ eachslice(Ω, dims=3)]
end