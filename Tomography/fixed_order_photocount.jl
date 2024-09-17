using BayesianTomography, HDF5, ProgressMeter

includet("../Utils/model_fitting.jl")
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/photocount_utils.jl")
##
calibration = h5open("Data/Raw/fixed_order_photocount.h5") do file
    read(file["calibration"])
end

x = LinRange(-0.5f0, 0.5f0, size(calibration, 1))
y = LinRange(-0.5f0, 0.5f0, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, maximum(calibration), minimum(calibration)])

fit_d = surface_fit(gaussian_model, x, y, calibration[:, :, 1], p0)
fit_c = surface_fit(gaussian_model, x, y, calibration[:, :, 2], p0)
##
order = 1
images, coefficients = h5open("Data/Raw/fixed_order_photocount.h5") do file
    read(file["images_order$order"]), read(file["labels_order$order"])
end

direct_operators = assemble_position_operators(x, y, fixed_order_basis(order, fit_d.param))
converted_basis = [(x, y) -> f(x, y) * cis((k - 1) * π / 2)
                   for (k, f) ∈ enumerate(fixed_order_basis(order, fit_c.param))]
converted_operators = assemble_position_operators(x, y, converted_basis)
operators = stack((direct_operators, converted_operators))

problem = StateTomographyProblem(operators)
mthd = MaximumLikelihood(problem)
##
m = 14
n = 2
outcomes = sample_events(view(images, :, :, :, m), photocounts[n])

sum(Int, outcomes)

ρ_pred, _ = prediction(outcomes, mthd, max_iter=10^2)
ψ = project2pure(ρ_pred)
fidelity(ψ, view(coefficients, :, m))
##
orders = 1:4
photocounts = [2^k for k ∈ 6:11]
fids = zeros(Float64, length(photocounts), 50, length(orders))

progress = Progress(length(fids))

for (k, order) ∈ enumerate(orders)
    images, coefficients = h5open("Data/Raw/fixed_order_photocount.h5") do file
        read(file["images_order$order"]), read(file["labels_order$order"])
    end

    direct_operators = assemble_position_operators(x, y, fixed_order_basis(order, fit_d.param))
    converted_basis = [(x, y) -> f(x, y) * cis((k - 1) * π / 2)
                       for (k, f) ∈ enumerate(fixed_order_basis(order, fit_c.param))]
    converted_operators = assemble_position_operators(x, y, converted_basis)
    operators = stack((direct_operators, converted_operators))

    problem = StateTomographyProblem(operators)
    mthd = BayesianInference(problem)


    Threads.@threads for m ∈ 1:50
        for n ∈ eachindex(photocounts)
            outcomes = sample_events(view(images, :, :, :, m), photocounts[n])
            ρ_pred, _ = prediction(outcomes, mthd)
            ψ = project2pure(ρ_pred)
            fids[n, m, k] = fidelity(ψ, view(coefficients, :, m))
            next!(progress)
        end
    end
end
finish!(progress)

dropdims(mean(fids, dims=2), dims=2)