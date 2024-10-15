using QuantumMeasurements, HDF5, ProgressMeter

includet("../Utils/basis.jl")
includet("../Utils/model_fitting.jl")
includet("../Utils/photocount_utils.jl")
includet("../Utils/bootstraping.jl")

path = "Data/fixed_order_photocount.h5"
##
calibration = h5open(path) do file
    file["calibration"] |> read
end

x = axes(calibration, 1)
y = axes(calibration, 2)
npixels = length(x) * length(y)
rs = Iterators.product(x, y)
sqrt_δA = sqrt((x[2] - x[1]) * (y[2] - y[1]) / 2) # We divide by 2 because we have two images

fit_d, fit_c = calibration_fit(x, y, calibration)

f!(buffer, r, pars) = fixed_order_basis!(buffer, r, pars)
g!(buffer, r, pars) = fixed_order_basis!(buffer, r, pars, Float32(π) / 2)

pars_d = (sqrt_δA, fit_d.param...)
pars_c = (sqrt_δA, fit_c.param...)
##
orders = 1:4
photocounts = [2^k for k ∈ 6:11]
fids = Array{Float32}(undef, length(photocounts), 50, length(orders))
mthd = MaximumLikelihood()

p = Progress(length(fids))
for (k, order) ∈ enumerate(orders)
    coefficients = h5open(path) do file
        file["labels_order$order"] |> read
    end

    μ = empty_measurement(2 * npixels, order + 1, Matrix{Float32})
    μ1 = view(μ, 1:npixels, :)
    μ2 = view(μ, npixels+1:2*npixels, :)
    buffer = Matrix{ComplexF32}(undef, order + 1, 512)

    multithreaded_update_measurement!(μ1, buffer, rs, pars_d, f!)
    multithreaded_update_measurement!(μ2, buffer, rs, pars_c, g!)


    Threads.@threads for n ∈ eachindex(photocounts)
        outcomes = h5open(path) do file
            file["images_order$order"] |> read
        end

        for m ∈ 1:50
            undersampled_image = sample_events(view(outcomes, :, :, :, m), photocounts[n])
            ψ = project2pure(estimate_state(undersampled_image, μ, mthd)[1])
            fids[n, m, k] = fidelity(ψ, view(coefficients, :, m))
            next!(p)
        end
    end
end

dropdims(mean(fids, dims=2), dims=2)
##
fids_no_calib = zeros(Float64, length(photocounts), 50, length(orders))
mthd = MaximumLikelihood()

p = Progress(length(fids_no_calib))
for (k, order) ∈ enumerate(orders)
    coefficients = h5open(path) do file
        file["labels_order$order"] |> read
    end

    Threads.@threads for n ∈ eachindex(photocounts)
        outcomes = h5open(path) do file
            file["images_order$order"] |> read
        end

        for m ∈ 1:50
            undersampled_image = sample_events(view(outcomes, :, :, :, m), photocounts[n])

            pars_d = (sqrt_δA, center_of_mass_and_waist(view(undersampled_image, :, :, 1), order)...)
            pars_c = (sqrt_δA, center_of_mass_and_waist(view(undersampled_image, :, :, 2), order)...)

            buffer = Matrix{ComplexF32}(undef, order + 1, 512)

            μ = empty_measurement(2 * npixels, order + 1, Matrix{Float32})
            μ1 = view(μ, 1:npixels, :)
            μ2 = view(μ, npixels+1:2*npixels, :)

            multithreaded_update_measurement!(μ1, buffer, rs, pars_d, f!)
            multithreaded_update_measurement!(μ2, buffer, rs, pars_c, g!)

            ψ = project2pure(estimate_state(undersampled_image, μ, mthd)[1])
            fids_no_calib[n, m, k] = fidelity(ψ, view(coefficients, :, m))
            next!(p)
        end
    end
end

dropdims(mean(fids_no_calib, dims=2), dims=2)

##
h5open("Results/fixed_order_photocount.h5", "cw") do file
    file["fid"] = stack(bootstrap(slice) for slice ∈ eachslice(fids, dims=(1, 3)))
    file["fid_no_calib"] = stack(bootstrap(slice) for slice ∈ eachslice(fids_no_calib, dims=(1, 3)))
    file["orders"] = collect(orders)
    file["photocounts"] = collect(photocounts)
end