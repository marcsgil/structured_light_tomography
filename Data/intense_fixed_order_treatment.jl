using BayesianTomography, HDF5, PositionMeasurements, ProgressMeter, LinearAlgebra
includet("../Utils/basis.jl")

file = h5open("Data/Processed/positive_l.h5")
fit_param = read(file["fit_param"])

rs = LinRange(-0.5, 0.5, 200)
##
dims = 6
fids = Matrix{Float64}(undef, length(dims), 100)

@showprogress for (m, dim) ∈ enumerate(dims)
    basis = positive_l_basis(dim, fit_param[1:4])
    povm = assemble_position_operators(rs, rs, basis)

    mthd = LinearInversion(povm)

    images = read(file["images_dim$dim"])
    ρs = read(file["labels_dim$dim"])

    for (n, probs) ∈ enumerate(eachslice(images, dims=3))
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