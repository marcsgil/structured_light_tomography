using CoherentNoise, Augmentor, Distributions,
    Tullio, LinearAlgebra, Images, HDF5, BayesianTomography

struct PureState end

function real_representation(ψs, ::PureState)
    #Represents the coeficients ψ as a real array.
    #We stack the real and then the imaginary part.
    D = size(ψs, 1)
    result = Array{real(eltype(ψs))}(undef, 2D, size(ψs, 2))
    result[1:D, :] = real.(@view ψs[1:end, :])
    result[D+1:2D, :] = imag.(@view ψs[1:end, :])
    result
end

function complex_representation(y, ::PureState)
    @. @views y[1:end÷2, :] + im * y[end÷2+1:end, :]
end

function generate_dataset!(dest, basis, ψs::AbstractMatrix)
    @tullio dest[x, y, n] = basis[x, y, r] * conj(basis[x, y, s]) * ψs[r, n] * ψs[s, n] |> real
end

function generate_dataset!(dest, basis, ρs)
    @tullio dest[x, y, n] = basis[x, y, r] * conj(basis[x, y, s]) * ρs[r, s, n] |> real
end


#Photocounting
function sample_photons!(images, N_photons, direct_prob=0.5)
    N = sum(images, dims=(1, 2))
    images ./= N
    images[:, :, 1, :] .*= direct_prob
    images[:, :, 2, :] .*= 1 - direct_prob

    Threads.@threads for image in eachslice(images, dims=4)
        simulate_outcomes!(image, N_photons)
    end
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
    augmentbatch!(CPUThreads(), x, x, pl)
end

function add_noise!(x; perlin_strength=0.3,
    squeezing_distribution=truncated(Normal(0.2, 0.1); lower=0),
    elastic_width=size(x, 1) ÷ 2, elastic_scale=0.1)

    if !isnothing(squeezing_distribution)
        Threads.@threads for slice ∈ eachslice(x, dims=3)
            squeeze!(slice, rand(squeezing_distribution))
        end
    end

    if !iszero(perlin_strength)
        Threads.@threads for image ∈ eachslice(x, dims=3)
            noise = perlin_noise_field(size(x, 1), size(x, 2), perlin_strength)
            image .*= noise
        end
    end

    if !iszero(elastic_width)
        elastic_distortion!(x; elastic_width, elastic_scale)
    end
end