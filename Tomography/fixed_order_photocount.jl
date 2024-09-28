using BayesianTomography, HDF5, ProgressMeter

includet("../Utils/position_operators.jl")
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
##
fit_d, fit_c = calibration_fit(x, y, calibration)

fit_c.param
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

    basis_d = fixed_order_basis(order, fit_d.param)
    basis_c = fixed_order_basis(order, fit_c.param, Float32(π) / 2)

    measurement = Measurement(assemble_position_operators(x, y, basis_d, basis_c))


    Threads.@threads for n ∈ eachindex(photocounts)
        outcomes = h5open(path) do file
            file["images_order$order"] |> read
        end

        for m ∈ 1:50
            undersampled_image = sample_events(view(outcomes, :, :, :, m), photocounts[n])
            ψ = project2pure(prediction(undersampled_image, measurement, mthd)[1])
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

            param_d = center_of_mass_and_waist(view(undersampled_image, :, :, 1), order)
            param_c = center_of_mass_and_waist(view(undersampled_image, :, :, 2), order)

            basis_d = fixed_order_basis(order, param_d)
            basis_c = fixed_order_basis(order, param_c, Float32(π) / 2)

            measurement = Measurement(assemble_position_operators(x, y, basis_d, basis_c))

            ψ = project2pure(prediction(undersampled_image, measurement, mthd)[1])
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