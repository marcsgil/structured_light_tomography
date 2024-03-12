using HDF5, CairoMakie, StatsBase, BayesianTomography, ProgressMeter

includet("../Data/data_treatment_utils.jl")
includet("../utils.jl")

input = h5open("Data/Processed/mixed_intense.h5")

direct_lims = read(input["direct_lims"])
converted_lims = read(input["converted_lims"])

xd, yd = get_grid(direct_lims, (400, 400))
xc, yc = get_grid(converted_lims, (400, 400))
weights = read(input["weights"])
##
orders = 1:5
fids = Vector{Float64}(undef, length(orders))
fids_std = Vector{Float64}(undef, length(orders))

for order ∈ orders
    images = read(input["images_order$order"])
    ρs = read(input["labels_order$order"])
    basis = transverse_basis(order)

    direct_operators = assemble_position_operators(xd, yd, basis)
    converted_operators = assemble_position_operators(xc, yc, basis)
    mode_converter = diagm([cis(-k * π / 6) for k ∈ 0:order])
    unitary_transform!(converted_operators, mode_converter)
    operators = compose_povm(direct_operators, converted_operators)
    mthd = LinearInversion(operators)

    this_fids = Vector{Float64}(undef, size(images, 4))

    Threads.@threads for n ∈ eachindex(this_fids)
        probs = view(images, :, :, :, n)
        σ = prediction(probs, mthd)
        this_fids[n] = fidelity(ρs[:, :, n], σ)
    end

    fids[order] = mean(this_fids)
    fids_std[order] = std(this_fids)
end

fids
##
out = h5open("New/Results/Intense/linear_inversion.h5", "w")
out["fids"] = fids
out["fids_std"] = fids_std
close(out)