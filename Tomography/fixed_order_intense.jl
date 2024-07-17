using BayesianTomography, HDF5, PositionMeasurements, ProgressMeter, LinearAlgebra
includet("../Utils/basis.jl")

file = h5open("Data/Processed/mixed_intense.h5")
direct_fit_param = read(file["direct_fit_param"])
converted_fit_param = read(file["converted_fit_param"])

rs = LinRange(-0.5, 0.5, 400)
##
orders = 1:5
fids = Matrix{Float64}(undef, length(orders), 100)

@showprogress for (m, order) ∈ enumerate(orders)
    direct_basis = fixed_order_basis(order, direct_fit_param[1:4]) |> reverse
    converted_basis = fixed_order_basis(order, converted_fit_param[1:4]) |> reverse
    direct_operators = assemble_position_operators(rs, rs, direct_basis)
    converted_operators = assemble_position_operators(rs, rs, converted_basis)
    mode_converter = diagm([cis(-k * π / 6) for k ∈ 0:order])
    unitary_transform!(converted_operators, mode_converter)
    povm = compose_povm(direct_operators, converted_operators)

    mthd = LinearInversion(povm)

    images = read(file["images_order$order"])
    ρs = read(file["labels_order$order"])

    for (n, probs) ∈ enumerate(eachslice(images, dims=4))
        σ = prediction(probs, mthd)
        fids[m, n] = fidelity(ρs[:, :, n], σ)
    end
end

dropdims(mean(fids, dims=2), dims=2)
##
out = h5open("New/Results/Intense/linear_inversion.h5", "w")
out["fids"] = dropdims(mean(fids, dims=2), dims=2)
out["fids_std"] = dropdims(std(fids, dims=2), dims=2)
close(out)