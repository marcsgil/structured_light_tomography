using BayesianTomography, HDF5, PositionMeasurements, ProgressMeter, LinearAlgebra
includet("../Utils/basis.jl")

file = h5open("Data/Processed/positive_l.h5")
fit_param = read(file["fit_param"])

rs = LinRange(-0.5, 0.5, 200)
##
using CairoMakie
visualize(file["images_dim2"][:, :, 3])
file["labels_dim6"][:, :, 8]
##
dims = 2:6
fids = Matrix{Float64}(undef, length(dims), 1)

@showprogress for (m, dim) ∈ enumerate(dims)
    basis = positive_l_basis(dim, fit_param[1:4])
    povm = assemble_position_operators(rs, rs, basis)

    mthd = LinearInversion(povm)

    images = file["images_dim$dim"][:,:,1:1]
    ρs = file["labels_dim$dim"][:,:,1]

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