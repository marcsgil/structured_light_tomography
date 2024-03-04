using ClassicalOrthogonalPolynomials, CoherentNoise,
    Augmentor, ProgressMeter, Distributions,
    Tullio, LinearAlgebra, Images,
    Plots, HDF5

include("samplers.jl")

#Basic functions

function hg(x, y, m, n)
    N = 1 / sqrt(2.0^(m + n) * factorial(m) * factorial(n) * pi)
    @tullio result[i, j] := (N * exp(-(x[i]^2 + y[j]^2) / 2)
                             * hermiteh(m, x[i]) * hermiteh(n, y[j]))
end

function get_basis(order, xs, ys; angle=π / 2)
    basis = Array{complex(eltype(xs))}(undef, length(xs), length(ys), 2, order + 1)

    for k in 0:order
        basis[:, :, 1, k+1] = hg(xs, ys, order - k, k)
        basis[:, :, 2, k+1] = hg(xs, ys, order - k, k) * cis(-angle * k)
    end

    basis
end

get_basis(order, rs; angle=π / 2) = get_basis(order, rs, rs; angle)

#Pure state representations

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

function generate_dataset(order, c, rs, ::PureState; angle=π / 2)
    basis = get_basis(order, rs; angle)

    #In one line we perform the superposition!! (Einstein summation convention)
    @tullio x[i, j, m, n] := basis[i, j, m, k] * c[k, n] |> abs2

    x, real_representation(c, PureState())
end

function generate_dataset(order, N_images::Integer, rs, ::PureState; angle=π / 2)
    c = sample_haar_vectors(order + 1, N_images)
    generate_dataset(order, c, rs, PureState(); angle)
end


#Mixed state representations

struct MixedState end

function A(dim, j)
    @assert dim > j "The dimension must be greater than the index."
    D = zeros(Float32, dim)
    for i ∈ 1:j
        D[i] = 1
    end
    D[j+1] = -j
    normalize!(D)
    diagm(D)
end

function B(dim, j, k)
    B = zeros(Float32, dim, dim)
    B[j, k] = 1
    B[k, j] = 1
    hermitianpart!(B)
    normalize!(B)
    B
end

function C(dim, j, k)
    C = zeros(ComplexF32, dim, dim)
    C[j, k] = im
    C[k, j] = -im
    hermitianpart!(C)
    normalize!(C)
    C
end

function get_basis(dim)
    As = stack(A(dim, j) for j ∈ 1:dim-1)
    Bs = Array{Float32}(undef, dim, dim, dim * (dim - 1) ÷ 2)
    Cs = Array{ComplexF32}(undef, dim, dim, dim * (dim - 1) ÷ 2)
    for j ∈ 2:dim, k ∈ 1:j-1
        Bs[:, :, (j-2)*(j-1)÷2+k] = B(dim, j, k)
        Cs[:, :, (j-2)*(j-1)÷2+k] = C(dim, j, k)
    end
    cat(As, Bs, Cs, dims=3)
end

function real_representation(ρs, ::MixedState)
    @assert size(ρs, 1) == size(ρs, 2) "The density matrices must be square."
    dim = size(ρs, 1)
    xs = Matrix{Float32}(undef, dim^2 - 1, size(ρs, 3))
    basis = get_basis(dim)
    for (ρ, x) ∈ zip(eachslice(ρs, dims=3), eachslice(xs, dims=2))
        for (i, b) ∈ enumerate(eachslice(basis, dims=3))
            x[i] = real(b ⋅ ρ)
        end
    end
    xs
end

function complex_representation(xs, ::MixedState)
    dim = Int(sqrt(size(xs, 1) + 1))
    basis = get_basis(dim)

    stack(I / dim + sum(c * b for (c, b) ∈ zip(x, eachslice(basis, dims=3))) for x ∈ eachslice(xs, dims=2))
end

function generate_dataset(order, ρs, rs, ::MixedState; angle=π / 2)
    basis = get_basis(order, rs; angle)

    @tullio x[i, j, m, n] := basis[i, j, m, k] * conj(basis[i, j, m, l]) * ρs[k, l, n] |> real

    x, real_representation(ρs, MixedState())
end

function generate_dataset(order, N_images::Integer, rs, ::MixedState; angle=π / 2)
    ρs = sample_from_product_measure(order + 1, N_images)
    generate_dataset(order, ρs, rs, MixedState(); angle)
end


#Photocounting

function array_representation(counts, imgsize)
    image = Array{Float32}(undef, imgsize...)
    Threads.@threads for n in eachindex(image)
        image[n] = count(x -> x == n, counts)
    end
    image
end

function sample_photons!(x, N_photons, direct_prob=0.5)
    N = sum(x, dims=(1, 2))
    x ./= N
    x[:, :, 1, :] .*= direct_prob
    x[:, :, 2, :] .*= 1 - direct_prob

    p = Progress(size(x, 4))
    Threads.@threads for slice in eachslice(x, dims=4)
        prob = normalize(vec(slice), 1)
        D = DiscreteNonParametric(eachindex(slice), prob)
        slice .= array_representation(rand(D, N_photons), size(slice))
        next!(p)
    end
    finish!(p)
end

#Noise
function perlin_noise_field(w, h, strength)
    x_0 = randn()
    y_0 = randn()
    noise = map(x -> x.r, gen_image(perlin_2d();
        w, h, xbounds=(x_0 - 2.0, x_0 + 2.0), ybounds=(y_0 - 2.0, y_0 + 2.0)))

    #Normalize the noise to be in the range [1-strength, 1+strength]
    m, M = extrema(noise)

    a = 2strength / (M - m)
    b = -a * m + 1 - strength
    @. noise = a * noise + b
    noise
end

function squeeze!(img, ratio)
    pad = zeros(eltype(img), size(img, 1), round(Int, size(img, 2) * abs(ratio) / 2))
    img .= imresize(hcat(pad, img, pad), size(img)...)
    nothing
end

function elastic_distortion!(x; elastic_width, elastic_scale=0.1)
    pl = ElasticDistortion(elastic_width; scale=elastic_scale)
    for slice ∈ eachslice(x, dims=3)
        augmentbatch!(CPUThreads(), slice, slice, pl)
    end
end

function add_noise!(x; perlin_strength=0.4,
    squeezing_distribution=truncated(Normal(0.3, 0.2); lower=0),
    elastic_width=15, elastic_scale=0.1)

    if !isnothing(squeezing_distribution)
        Threads.@threads for slice ∈ eachslice(x, dims=4)
            squeeze!(view(slice, :, :, 2), rand(squeezing_distribution))
        end
    end

    if !iszero(perlin_strength)
        Threads.@threads for image ∈ eachslice(x, dims=(3, 4))
            noise = perlin_noise_field(size(x, 1), size(x, 2), perlin_strength)
            image .*= noise
        end
    end

    if !iszero(elastic_width)
        elastic_distortion!(x; elastic_width, elastic_scale)
    end
end
##
order = 1
rs = LinRange(-3.0f0, 3.0f0, 64)
x, y = generate_dataset(order, 10, rs, MixedState(), angle=π / 6)

#add_noise!(x, perlin_strength=0.05, elastic_scale=0.08)
add_noise!(x)
heatmap(x[:, :, 1, 10])

sample_photons!(x, 2048)
heatmap(x[:, :, 2, 10])
sum(x[:, :, :, 1])
##
nobs = 2 .^ (6:11)

file = h5open("Data/TrainingData/intense_mixed.h5", "cw")
@showprogress for order ∈ 1:5
    x, y = nothing, nothing
    GC.gc()
    R = 2.5f0 + 0.5f0 * order
    rs = LinRange(-R, R, 64)
    x, y = generate_dataset(order, 10^5, rs, MixedState(), angle=π / 6)
    add_noise!(x)
    #sample_photons!(x, n)

    file["x_order$(order)"] = x
    file["y_order$(order)"] = y
end
close(file)
##
using HDF5

file = h5open("ExperimentalData/Intense/mixed.h5", "cw")
for order ∈ 2:5
    ρs = file["labels_order$(order)"][:, :, :]
    ys = real_representation(ρs, MixedState())
    file["ys_order$(order)"] = ys
end
close(file)