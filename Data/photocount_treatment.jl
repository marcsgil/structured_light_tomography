using TiffImages, ProgressMeter, HDF5

function ncounts(x, μ, σ)
    n = (x - μ) / 5σ
    max(zero(n), floor(n))
end

function add_photocounts!(dest, image::AbstractMatrix, μ, σ)
    for n ∈ eachindex(dest)
        dest[n] += ncounts(image[n], μ[n], σ[n])
    end
end

function add_photocounts!(dest, image, μ, σ)
    for slice ∈ eachslice(image, dims=3)
        add_photocounts!(dest, slice, μ, σ)
    end
end

function get_photocounts(paths, μ, σ, ::Type{T}=UInt8) where {T}
    dest = zeros(T, 64, 64, 2, length(paths))

    Threads.@threads for m ∈ eachindex(paths)
        image = real.(TiffImages.load(paths[m], verbose=false, mmap=true))
        add_photocounts!(view(dest, :, :, :, m), image, μ, σ)
    end

    reverse!(dest, dims=3) #So that the direct image comes first
end
##
background = real.(TiffImages.load("Data/Raw/Photocount/background.tif"))
μ = reshape(mean(background, dims=3), (64, 64, 2))
σ = reshape(std(background, dims=3), (64, 64, 2))
reverse!(μ, dims=3) #So that the direct image comes first
reverse!(σ, dims=3) #So that the direct image comes first
##

h5open("Data/Raw/fixed_order_photocount.h5", "cw") do file
    @showprogress for order ∈ 1:4
        paths = ["Data/Raw/Photocount/Order$order/$n.tif" for n ∈ 0:49]
        file["images_order$order"] = get_photocounts(paths, μ, σ)
        file["labels_order$order"] = h5open("Data/Raw/Photocount/coefficients.h5") do file_coeff
            # The coefficients were stored with a different basis ordering
            reverse(read(file_coeff["order$order"]), dims=1)
        end
    end

    calibration = reshape(real.(TiffImages.load("Data/Raw/Photocount/calibration.tif")), (64, 64, 2))
    reverse!(calibration, dims=3) #So that the direct image comes first
    calibration = reinterpret(UInt16, calibration)

    file["calibration"] = calibration
end