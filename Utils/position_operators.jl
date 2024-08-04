function assemble_position_operators(xs, ys, basis)
    T = complex(float(eltype(xs)))
    Π = Matrix{T}(undef, length(basis), length(basis))
    operators = Matrix{Matrix{T}}(undef, length(xs), length(ys))

    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]
    ΔA = Δx * Δy


    """for (n, y) ∈ enumerate(ys), (m, x) ∈ enumerate(xs)
        for (k, ψ) ∈ enumerate(basis), j ∈ eachindex(basis)
            ϕ = basis[j]
            Π[j, k] = conj(ϕ(x, y)) * ψ(x, y) * ΔA
        end
        operators[m, n] = copy(Π)
    end

    Ns = sqrt.(sum(diag, operators))

    for Π ∈ operators
        for n ∈ axes(Π, 2), m ∈ axes(Π, 1)
            Π[m, n] /= (Ns[m] * Ns[n])
        end
    end"""

    basis = [ϕ(x, y) for x ∈ xs, y ∈ ys, ϕ ∈ basis]

    for s ∈ eachslice(basis, dims=3)
        N = sum(abs2, s)
        s ./= √(N * ΔA)
    end

    operators = map(eachslice(basis, dims=(1, 2))) do v
        conj.((v * v'))
    end

    return operators
end