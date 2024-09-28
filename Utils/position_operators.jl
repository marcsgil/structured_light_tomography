using Tullio, LinearAlgebra

function assemble_position_operators(xs, ys, basis...)
    Δx = xs[2] - xs[1]
    Δy = ys[2] - ys[1]
    normalization = sqrt(Δx * Δy / length(basis))

    operators = Array{ComplexF32}(undef, length(first(basis)), length(xs), length(ys), length(basis))

    for (k, base) ∈ enumerate(basis)
        for (j, f) ∈ enumerate(base)
            operators[j, :, :, k] = f(xs, ys, normalization)
        end
    end

    operators
    eachslice(operators, dims=(2, 3, 4))
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