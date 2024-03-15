using HDF5, StatsBase, BayesianTomography, PositionMeasurements, ProgressMeter

includet("../Data/data_treatment_utils.jl")

input = h5open("Data/Processed/mixed_intense.h5")

direct_lims = read(input["direct_lims"])
converted_lims = read(input["converted_lims"])

xd, yd = get_grid(direct_lims, (400, 400))
xc, yc = get_grid(converted_lims, (400, 400))
weights = read(input["weights"])
##
orders = 1:5
fids = Matrix{Float64}(undef, length(orders), 100)

@showprogress for (m, order) ∈ enumerate(orders)
    images = read(input["images_order$order"])
    ρs = read(input["labels_order$order"])
    basis = transverse_basis(order)

    direct_operators = assemble_position_operators(xd, yd, basis)
    converted_operators = assemble_position_operators(xc, yc, basis)
    mode_converter = diagm([cis(-k * π / 6) for k ∈ 0:order])
    unitary_transform!(converted_operators, mode_converter)
    operators = compose_povm(direct_operators, converted_operators)
    mthd = LinearInversion(operators)

    for (n, probs) ∈ enumerate(eachslice(images, dims=4))
        σ = prediction(probs, mthd)
        fids[m, n] = fidelity(ρs[:, :, n], σ)
    end
end

dropdims(mean(fids, dims=2), dims=2)
##
out = h5open("New/Results/Intense/linear_inversion.h5", "w")
out["fids"] = fids
out["fids_std"] = fids_std
close(out)