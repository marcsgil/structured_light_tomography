using TiffImages, ProgressMeter, HDF5

includet("../data_treatment_utils.jl")

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
            slice = reverse(slice, dims=3) # So that the direct image to come first
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
background = real.(TiffImages.load("New/Data/Photocount/Raw/background.tif"))
μ = reshape(mean(background, dims=3), (64, 64, 2))
σ = reshape(std(background, dims=3), (64, 64, 2))
reverse!(μ, dims=3) #So that the direct image comes first
reverse!(σ, dims=3) #So that the direct image comes first
## 

out = h5open("New/Data/Photocount/data.h5", "cw")

for order in 1:4
    histories = extract_histories(["New/Data/Photocount/Raw/Order$order/$n.tif" for n ∈ 0:49], 2048, μ, σ)
    out["histories_order$order"] = histories
end

calibration = reshape(real.(TiffImages.load("New/Data/Photocount/Raw/calibration.tif")), (64, 64, 2))
reverse!(calibration, dims=3) #So that the direct image comes first

out["calibration"] = reinterpret(UInt16, calibration)

coeffs = h5open("New/Data/Photocount/Raw/coefficients.h5")

for order in 1:4
    out["coefficients_order$order"] = read(coeffs["order$order"])
end

close(coeffs)

close(out)