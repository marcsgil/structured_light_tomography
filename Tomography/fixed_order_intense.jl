using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra, FiniteDiff
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/model_fitting.jl")
includet("../Data/data_treatment_utils.jl")
includet("../Utils/metrics.jl")

input = h5open("Data/Processed/mixed_intense.h5")

direct_lims = read(input["direct_lims"])
converted_lims = read(input["converted_lims"])

xd, yd = get_grid(direct_lims, (400, 400))
xc, yc = get_grid(converted_lims, (400, 400))
weights = read(input["weights"])
##
orders = 1:5
metrics = Matrix{Float64}(undef, length(orders), 100)
errors = Matrix{Float64}(undef, length(orders), 100)

@showprogress for (m, order) ∈ enumerate(orders)
    images = read(input["images_order$order"])
    ρs = read(input["labels_order$order"])
    basis = fixed_order_basis(order, [0, 0, √2, 1])

    direct_povm = assemble_position_operators(xd, yd, basis)
    converted_povm = assemble_position_operators(xc, yc, basis)
    mode_converter = diagm([cis(-k * π / 6) for k ∈ 0:order])
    unitary_transform!(converted_povm, mode_converter)
    povm = compose_povm(direct_povm, converted_povm)
    problem = StateTomographyProblem(povm)
    mthd = LinearInversion(problem)

    Threads.@threads for n ∈ axes(images, 4)
        probs = @view images[:, :, :, n]
        ρ = @view ρs[:, :, n]
        θs = gell_mann_projection(ρ)
        pred_ρ, pred_θs, cov = prediction(probs, mthd)
        
        metrics[m, n], errors[m,n] = fidelity_metric(θs, pred_θs, cov, 0.95)
    end
end

dropdims(mean(metrics, dims=2), dims=2)
dropdims(mean(errors, dims=2), dims=2)
##
out = h5open("New/Results/Intense/linear_inversion.h5", "w")
out["fids"] = dropdims(mean(fids, dims=2), dims=2)
out["fids_std"] = dropdims(std(fids, dims=2), dims=2)
close(out)