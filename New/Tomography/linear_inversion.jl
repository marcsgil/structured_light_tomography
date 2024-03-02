using HDF5, CairoMakie, StatsBase, BayesianTomography, ProgressMeter

includet("../Data/data_treatment_utils.jl")
includet("../utils.jl")

input = h5open("New/Data/Intense/mixed.h5")
calibration = read(input["calibration"])

remove_background!(calibration)
direct_lims, converted_lims = get_limits(calibration)
xd, yd = get_grid(direct_lims, size(calibration))
xc, yc = get_grid(converted_lims, size(calibration))
##
orders = 1:5
fids = Vector{Float64}(undef, length(orders))
fids_std = Vector{Float64}(undef, length(orders))

for order ∈ orders
    images = read(input["images_order$order"])
    remove_background!(images)
    ρs = read(input["labels_order$order"])
    basis = [(r, par) -> hg(r[1], r[2], order - m, m) for m ∈ 0:order]

    direct_operators = assemble_position_operators(xd, yd, basis)
    converted_operators = assemble_position_operators(xc, yc, basis)
    mode_converter = diagm([cis(-k * π / 6) for k ∈ 0:order])
    unitary_transform!(converted_operators, mode_converter)
    operators = compose_povm(direct_operators, converted_operators)
    mthd = LinearInversion(operators)

    this_fids = Vector{Float64}(undef, size(images, 4))

    Threads.@threads for n ∈ eachindex(this_fids)
        direct = normalize(images[:, :, 1, n], 1)
        converted = normalize(images[:, :, 2, n], 1)
        probs = cat(direct / 2, converted / 2, dims=3)
        σ = project2density(prediction(probs, mthd))
        this_fids[n] = fidelity(ρs[:, :, n], σ)
    end

    fids[order] = mean(this_fids)
    fids_std[order] = std(this_fids)
end

fids
fids_std
##
out = h5open("New/Results/Intense/linear_inversion.h5", "w")
out["fids"] = fids
out["fids_std"] = fids_std
close(out)