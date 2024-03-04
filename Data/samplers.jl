using LinearAlgebra

function sample_haar_unitaries(dim, n_samples)
    Zs = randn(ComplexF32, dim, dim, n_samples)

    for Z ∈ eachslice(Zs, dims=3)
        Q, R = qr(Z)
        Λ = diag(R)
        @. Λ /= abs(Λ)
        Z .= Q * diagm(Λ)
    end

    return Zs
end

function sample_haar_vectors(dim, n_samples)
    sample_haar_unitaries(dim, n_samples)[1, :, :]
end

function sample_from_simplex(dim)
    ξs = rand(Float32, dim - 1)
    λs = Vector{Float32}(undef, dim)
    for (k, ξ) ∈ enumerate(ξs)
        λs[k] = (1 - ξ^(1 / (dim - k))) * (1 - sum(λs[1:k-1]))
    end
    λs[end] = 1 - sum(λs[1:end-1])
    λs
end

function sample_from_simplex(dim, nsamples)
    stack(sample_from_simplex(dim) for _ ∈ 1:nsamples)
end


function combine!(unitaries::Array{T1,3}, probabilities::Array{T2,2}) where {T1,T2}
    @assert size(probabilities, 2) == size(unitaries, 3) "The number of probabilities and unitaries must be the same."
    @assert size(probabilities, 1) == size(unitaries, 1) "The dimension of the probabilities and unitaries must be the same."

    for (U, p) ∈ zip(eachslice(unitaries, dims=3), eachslice(probabilities, dims=2))
        U .= U * diagm(p) * U'
    end
end

function sample_from_product_measure(dims, n_samples)
    ps = sample_from_simplex(dims, n_samples)
    Us = sample_haar_unitaries(dims, n_samples)
    combine!(Us, ps)
    Us
end

function sample_from_ginibre_ensemble(dim, n_samples)
    ρs = randn(ComplexF32, dim, dim, n_samples)

    for ρ ∈ eachslice(ρs, dims=3)
        ρ .= ρ * ρ'
        ρ ./= tr(ρ)
    end

    ρs
end