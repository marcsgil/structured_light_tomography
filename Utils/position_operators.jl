using Tullio, LinearAlgebra

function assemble_position_operators(xs, ys, basis)
    T = complex(float(eltype(xs)))
    Π = Matrix{T}(undef, length(basis), length(basis))
    operators = Matrix{Matrix{T}}(undef, length(xs), length(ys))

    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]
    ΔA = Δx * Δy

    for (n, y) ∈ enumerate(ys), (m, x) ∈ enumerate(xs)
        for (k, ψ) ∈ enumerate(basis), (j, ϕ) ∈ enumerate(basis)
            Π[j, k] = conj(ϕ(x, y)) * ψ(x, y) * ΔA
        end
        operators[m, n] = copy(Π)
    end

    return operators


    """basis_fields = stack(basis(xs, ys) for basis ∈ basis)
    broadcast!(conj, basis_fields, basis_fields)
    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]
    ΔA = Δx * Δy

    [v * v' * ΔA for v ∈ eachslice(basis_fields, dims=(1, 2))]"""
end

function get_intensity!(img, buffer, ρ, basis_func, x, y)
    for (n, y) ∈ enumerate(y), (m, x) ∈ enumerate(x)
        for (i, f) ∈ enumerate(basis_func)
            buffer[i] = f(x, y)
        end
        img[m, n] = real(dot(buffer, ρ, buffer))
    end
end

function get_intensity(ρ, basis_func, x, y)
    img = Matrix{Float32}(undef, length(x), length(y))
    buffer = [f(x[1], y[1]) for f in basis_func]
    get_intensity!(img, buffer, ρ, basis_func, x, y)
    img
end