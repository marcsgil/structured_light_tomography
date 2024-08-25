using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/model_fitting.jl")
includet("../Data/data_treatment_utils.jl")

input = h5open("Data/Processed/mixed_intense.h5")

direct_lims = read(input["direct_lims"])
converted_lims = read(input["converted_lims"])

xd, yd = get_grid(direct_lims, (400, 400))
xc, yc = get_grid(converted_lims, (400, 400))
weights = read(input["weights"])
##
basis = fixed_order_basis(1, [0, 0, √2, 1])

direct_povm = assemble_position_operators(xd, yd, basis)
converted_povm = assemble_position_operators(xc, yc, basis)
mode_converter = diagm([cis(-k * π / 2) for k ∈ 0:1])
unitary_transform!(converted_povm, mode_converter)
povm = compose_povm(direct_povm, converted_povm)
problem = StateTomographyProblem(povm)


eigvals(fisher(problem, [0, 0, 0.]))

sum(inv, eigvals(fisher(problem, [0, 0, 0.])))
##
orders = 1:5
fids = Matrix{Float64}(undef, length(orders), 100)

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

    for n ∈ axes(images, 4)
        probs = vec(images[:, :, :, n])
        σ, _ = prediction(probs, mthd)
        σ = project2density(σ)
        fids[m, n] = fidelity(ρs[:, :, n], σ)
        #fids[m, n] = real(tr((ρs[:, :, n] - σ)^2))
    end
end

dropdims(mean(fids, dims=2), dims=2)
##
out = h5open("New/Results/Intense/linear_inversion.h5", "w")
out["fids"] = dropdims(mean(fids, dims=2), dims=2)
out["fids_std"] = dropdims(std(fids, dims=2), dims=2)
close(out)