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

"""function fisher_information!(dest, probs::AbstractVector, C::AbstractMatrix)
    for I ∈ eachindex(IndexCartesian(), dest)
        tmp = zero(eltype(dest))
        for k ∈ eachindex(probs)
            tmp += C[I[1], k] * C[I[2], k] / probs[k]
        end
        dest[I] = tmp
    end
end

function fisher_information(ρ, povm)
    probs = vec([real(ρ ⋅ E) for E in povm])
    basis = gell_mann_matrices(size(ρ, 1), include_identity=false)
    C = hcat((real_orthogonal_projection(Π, basis) for Π in povm)...)
    I = similar(C, size(C, 1), size(C, 1))
    fisher_information!(I, probs, C)
    I
end"""