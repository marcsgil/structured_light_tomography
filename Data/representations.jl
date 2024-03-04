using BayesianTomography, LinearAlgebra

struct PureState end

function real_representation(c, ::PureState)
    #Represents the coeficients c as a real array.
    #We stack the real and then the imaginary part.
    D = size(c, 1)
    result = Array{real(eltype(c))}(undef, 2D, size(c, 2))
    result[1:D, :] = real.(@view c[1:end, :])
    result[D+1:2D, :] = imag.(@view c[1:end, :])
    result
end

function complex_representation(y, ::PureState)
    @views y[1:end÷2, :] + im * y[end÷2+1:end, :]
end

struct MixedState end

function real_representation(ρs, ::MixedState)
    @assert size(ρs, 1) == size(ρs, 2) "The density matrices must be square."
    dim = size(ρs, 1)
    xs = Matrix{Float32}(undef, dim^2 - 1, size(ρs, 3))
    basis = get_hermitian_basis(dim)
    for (ρ, x) ∈ zip(eachslice(ρs, dims=3), eachslice(xs, dims=2))
        for n ∈ 2:dim^2
            x[n-1] = real(basis[n] ⋅ ρ)
        end
    end
    xs
end

function complex_representation(xs, ::MixedState)
    dim = Int(sqrt(size(xs, 1) + 1))
    basis = get_hermitian_basis(dim)

    stack(I / dim + sum(x[n] * basis[n+1] for n ∈ eachindex(x)) for x ∈ eachslice(xs, dims=2))
end