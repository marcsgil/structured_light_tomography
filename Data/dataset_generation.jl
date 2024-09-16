using CoherentNoise, Augmentor, Distributions,
    Tullio, LinearAlgebra, Images, BayesianTomography, ProgressMeter

function generate_dataset!(dest, basis, ψs::AbstractMatrix)
    @tullio dest[x, y, n] = basis[x, y, r] * conj(basis[x, y, s]) * ψs[r, n] * conj(ψs[s, n]) |> real
end

function generate_dataset!(dest, basis, ρs)
    @tullio dest[x, y, n] = basis[x, y, r] * conj(basis[x, y, s]) * ρs[r, s, n] |> real
end

#Photocounting
function sample_photons!(images, N_photons; dims=3)
    N = sum(images, dims=Tuple(i for i in 1:ndims(arr) if i ∉ dims))
    images ./= N

    Threads.@threads for image in eachslice(images; dims)
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
    shear_angles=-15:5:15,
    elastic_width=size(x, 1) ÷ 3, elastic_scale=0.1, dims=3)

    pipeline = (
        ShearX(shear_angles) * ShearY(shear_angles) * NoOp()
        |> CropNative((axes(x, 1), axes(x, 2)))
        |> ElasticDistortion(elastic_width; scale=elastic_scale)
    )

    p = Progress(prod(size(x, dim) for dim ∈ dims))
    Threads.@threads for slice ∈ eachslice(x; dims)
        img = copy(slice)
        noise = perlin_noise_field(size(x, 1), size(x, 2), perlin_strength)
        img .*= noise
        augment!(slice, img, pipeline)
        next!(p)
    end
    finish!(p)
end