using TiffImages, ProgressMeter, HDF5, BayesianTomography

include("data_treatment_utils.jl")

ncounts(x, μ, σ) = floor(Int64, (x - μ) / 5σ)

function extract_histories(paths, L, μ, σ)
    histories = Matrix{Int64}(undef, L, length(paths))

    p = Progress(length(paths), 1)
    Threads.@threads for m ∈ eachindex(paths)
        data = TiffImages.load(paths[m], verbose=false, mmap=true)

        counter = 0

        for _slice ∈ eachslice(data, dims=3)
            if counter > L
                break
            end
            slice = reshape(_slice, size(_slice, 1), size(_slice, 2) ÷ 2, 2)
            slice = reverse(slice, dims=3) # So that the direct image comes first
            for n ∈ eachindex(slice)
                if counter > L
                    break
                end
                for _ ∈ 1:ncounts(real(slice[n]), μ[n], σ[n])
                    counter += 1
                    if counter > L
                        break
                    end
                    histories[counter, m] = n
                end
            end
        end
        next!(p)
    end
    finish!(p)
    histories
end
##
background = real.(TiffImages.load("Data/Raw/Photocount/background.tif"))
μ = reshape(mean(background, dims=3), (64, 64, 2))
σ = reshape(std(background, dims=3), (64, 64, 2))
reverse!(μ, dims=3) #So that the direct image comes first
reverse!(σ, dims=3) #So that the direct image comes first
coeffs = h5open("Data/Raw/Photocount/coefficients.h5")
## 
out = h5open("Data/Processed/pure_photocount.h5", "cw")
photocounts = [2^k for k ∈ 6:11]

for order in 1:4
    histories = extract_histories(["Data/Raw/Photocount/Order$order/$n.tif" for n ∈ 0:49], 2048, μ, σ)
    out["histories_order$order"] = histories

    # Looks like the coefficients are stored in the reverse order
    out["labels_order$order"] = reverse(read(coeffs["order$order"]), dims=1)
    for pc ∈ photocounts
        images = Array{UInt8}(undef, (64, 64, 2, size(histories, 2)))
        for n ∈ axes(histories, 2)
            images[:, :, :, n] = history2array(histories[1:pc, n], (64, 64, 2)) .|> UInt8
        end
        out["images_order$order/$(pc)_photocounts"] = images
    end
end

calibration = reshape(real.(TiffImages.load("Data/Raw/Photocount/calibration.tif")), (64, 64, 2))
reverse!(calibration, dims=3) #So that the direct image comes first
calibration = reinterpret(UInt16, calibration)
remove_background!(calibration)
direct_lims, converted_lims = get_limits(calibration)

out["direct_lims"] = direct_lims
out["converted_lims"] = converted_lims

out["weights"] = [sum(calibration[:, :, 1]), sum(calibration[:, :, 2])] / sum(calibration)

close(coeffs)
close(out)