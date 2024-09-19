using BayesianTomography, HDF5, ProgressMeter

includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("../Utils/model_fitting.jl")
includet("../Utils/photocount_utils.jl")

path = "Data/Raw/fixed_order_photocount.h5"
##
calibration = h5open(path) do file
    file["calibration"] |> read
end

x = LinRange(-0.5f0, 0.5f0, size(calibration, 1))
y = LinRange(-0.5f0, 0.5f0, size(calibration, 2))

fit_d, fit_c = calibration_fit(x, y, calibration)
##
orders = 1:4
photocounts = [2^k for k ∈ 6:11]
fids = zeros(Float64, length(photocounts), 50, length(orders))

p = Progress(length(fids))
for (k, order) ∈ enumerate(orders)
    coefficients = h5open(path) do file
        file["labels_order$order"] |> read
    end

    basis_d = fixed_order_basis(order, fit_d.param)
    basis_c = [(x, y) -> f(x, y) * cis((k - 1) * Float32(π) / 2)
               for (k, f) ∈ enumerate(fixed_order_basis(order, fit_c.param))]

    direct_povm = assemble_position_operators(x, y, basis_d)
    converted_povm = assemble_position_operators(x, y, basis_c)
    povm = stack((direct_povm, converted_povm))
    problem = StateTomographyProblem(povm)
    mthd = MaximumLikelihood(problem)

    Threads.@threads for n ∈ eachindex(photocounts)
        outcomes = h5open(path) do file
            file["images_order$order"] |> read
        end

        for m ∈ 1:50
            undersampled_image = sample_events(view(outcomes, :, :, :, m), photocounts[n])
            ψ = project2pure(prediction(undersampled_image, mthd)[1])
            fids[n, m, k] = fidelity(ψ, view(coefficients, :, m))
            next!(p)
        end
    end
end

fids = dropdims(mean(fids, dims=2), dims=2)