using BayesianTomography, HDF5, ProgressMeter, LuxUtils, LuxCUDA

includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("../Utils/model_fitting.jl")
includet("../Utils/photocount_utils.jl")

path = "Data/Raw/fixed_order_photocount.h5"

"""model = get_model()
ps, st = jldopen("Tomography/TrainingLogs/best_model_pc.jld2") do file
    file["parameters"], file["states"]
end |> gpu_device()"""
##
calibration = h5open(path) do file
    file["calibration"] |> read
end

"""x = LinRange(-0.5f0, 0.5f0, size(calibration, 1))
y = LinRange(-0.5f0, 0.5f0, size(calibration, 2))"""

x = axes(calibration, 1)
y = axes(calibration, 2)

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

    for n ∈ eachindex(photocounts)
        outcomes = h5open(path) do file
            file["images_order$order"] |> read
        end

        undersampled_outcomes = stack(
            sample_events(slice, photocounts[n]) .|> Float32 for slice ∈ eachslice(outcomes, dims=4)
        ) #|> gpu_device()

        #param_d = model(view(undersampled_outcomes, :, :, 1:1, :), ps, st)[1] |> cpu_device()
        #param_c = model(view(undersampled_outcomes, :, :, 2:2, :), ps, st)[1] |> cpu_device()

        #undersampled_outcomes = undersampled_outcomes |> cpu_device()

        Threads.@threads for m ∈ 1:50
            param_d = center_of_mass_and_waist(view(undersampled_outcomes, :, :, 1, m), order - 1)
            param_c = center_of_mass_and_waist(view(undersampled_outcomes, :, :, 2, m), order - 1)

            basis_d = fixed_order_basis(order, param_d)
            basis_c = [(x, y) -> f(x, y) * cis((k - 1) * Float32(π) / 2)
                       for (k, f) ∈ enumerate(fixed_order_basis(order, param_c))]

            direct_povm = assemble_position_operators(x, y, basis_d)
            converted_povm = assemble_position_operators(x, y, basis_c)
            povm = stack((direct_povm, converted_povm))
            problem = StateTomographyProblem(povm)
            mthd = MaximumLikelihood(problem)

            ψ = project2pure(prediction(view(undersampled_outcomes, :, :, :, m), mthd)[1])
            fids[n, m, k] = fidelity(ψ, view(coefficients, :, m))
            next!(p)
        end
    end
end

fids = dropdims(mean(fids, dims=2), dims=2)