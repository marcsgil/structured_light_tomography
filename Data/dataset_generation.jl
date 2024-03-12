using CoherentNoise, Augmentor, Distributions,
    Tullio, LinearAlgebra, Images, HDF5, BayesianTomography

includet("samplers.jl")
includet("representations.jl")
includet("../utils.jl")

#Pure state representations

function generate_dataset(c, rs, angle, ::PureState)
    basis = transverse_basis(rs, rs, rs, rs, size(c, 1) - 1, angle)

    x = Array{Float32}(undef, length(rs), length(rs), 2, size(c, 2))

    Threads.@threads for k ∈ axes(c, 2)
        label2image!(view(x, :, :, :, k), view(c, :, k), basis)
    end

    x, real_representation(c, PureState())
end

function generate_dataset(order, N_images, rs, angle, ::PureState)
    c = sample_haar_vectors(order + 1, N_images)
    generate_dataset(c, rs, angle, PureState())
end

function generate_dataset(ρs, rs, angle, ::MixedState)
    basis = transverse_basis(rs, rs, rs, rs, size(ρs, 1) - 1, angle)

    x = Array{Float32}(undef, length(rs), length(rs), 2, size(ρs, 3))

    Threads.@threads for k ∈ axes(ρs, 3)
        label2image!(view(x, :, :, :, k), view(ρs, :, :, k), basis)
    end

    x, real_representation(ρs, MixedState())
end

function generate_dataset(order, N_images, rs, angle, ::MixedState)
    ρs = sample_from_product_measure(order + 1, N_images)
    generate_dataset(ρs, rs, angle, MixedState())
end

#Photocounting
function sample_photons!(images, N_photons, direct_prob=0.5)
    N = sum(images, dims=(1, 2))
    images ./= N
    images[:, :, 1, :] .*= direct_prob
    images[:, :, 2, :] .*= 1 - direct_prob

    p = Progress(size(images, 4))
    Threads.@threads for image in eachslice(images, dims=4)
        simulate_outcomes!(image, N_photons)
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

function horizontal_squeeze!(img, ratio1, ratio2)
    pad1 = zeros(eltype(img), size(img, 1), round(Int, size(img, 2) * abs(ratio1)))
    pad2 = zeros(eltype(img), size(img, 1), round(Int, size(img, 2) * abs(ratio2)))
    img .= imresize(hcat(pad1, img, pad2), size(img)...)
    nothing
end

function vertical_squeeze!(img, ratio1, ratio2)
    pad1 = zeros(eltype(img), round(Int, size(img, 1) * abs(ratio1)), size(img, 2))
    pad2 = zeros(eltype(img), round(Int, size(img, 1) * abs(ratio2)), size(img, 2))
    img .= imresize(vcat(pad1, img, pad2), size(img)...)
    nothing
end

function squeeze!(img, ratio1, ratio2)
    rand() > 0.5 ? horizontal_squeeze!(img, ratio1, ratio2) : vertical_squeeze!(img, ratio1, ratio2)
end

squeeze!(img, ratio) = squeeze!(img, ratio, ratio)

function elastic_distortion!(x; elastic_width, elastic_scale=0.1)
    pl = ElasticDistortion(elastic_width; scale=elastic_scale)
    for slice ∈ eachslice(x, dims=3)
        augmentbatch!(CPUThreads(), slice, slice, pl)
    end
end

function add_noise!(x; perlin_strength=0.3,
    squeezing_distribution=truncated(Normal(0.2, 0.1); lower=0),
    elastic_width=size(x, 1) ÷ 2, elastic_scale=0.1)

    if !isnothing(squeezing_distribution)
        Threads.@threads for slice ∈ eachslice(x, dims=(3, 4))
            squeeze!(slice, rand(squeezing_distribution))
        end
    end

    if !iszero(elastic_width)
        elastic_distortion!(x; elastic_width, elastic_scale)
    end

    if !iszero(perlin_strength)
        Threads.@threads for image ∈ eachslice(x, dims=(3, 4))
            noise = perlin_noise_field(size(x, 1), size(x, 2), perlin_strength)
            image .*= noise
        end
    end
end