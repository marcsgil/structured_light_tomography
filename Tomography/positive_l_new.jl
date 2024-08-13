using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra, CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")

relu(x, y) = x > y ? x - y : zero(x)

file = h5open("Data/Raw/postive_l_new.h5")
fit_param = attrs(file["calibration"])["fit_param"]

fit_param

rs = LinRange(-0.5, 0.5, 200)
##
dims = 2:6
fids = Matrix{Float64}(undef, length(dims), 100)

@showprogress for (m, dim) ∈ enumerate(dims)
    basis = positive_l_basis(dim, fit_param[1:4])
    povm = assemble_position_operators(rs, rs, basis)

    mthd = LinearInversion(povm)

    images = file["images_dim$dim"][:, :, :]
    ρs = attrs(file["images_dim$dim"])["density_matrices"]
    map!(x -> relu(x, 0x03), images, images)

    for (n, probs) ∈ enumerate(eachslice(images, dims=3))
        σ, _ = prediction(probs, mthd)
        #fids[m, n] = fidelity(ρs[:, :, n], σ)
        fids[m, n] = sum(abs2, ρs[:, :, n] .- σ)
    end
end

dropdims(mean(fids, dims=2), dims=2)
##
out = h5open("New/Results/Intense/linear_inversion.h5", "w")
out["fids"] = dropdims(mean(fids, dims=2), dims=2)
out["fids_std"] = dropdims(std(fids, dims=2), dims=2)
close(out)
##
dim = 2
basis = positive_l_basis(dim, fit_param[1:4])
povm = assemble_position_operators(rs, rs, basis)

mthd = LinearInversion(povm)

n = 1
img = file["images_dim$dim"][:, :, n]
ρ = attrs(file["images_dim$dim"])["density_matrices"][:,:,n]

σ, _ = prediction(img, mthd)

σ
ρ

theo = [real(tr(Π * ρ)) for Π ∈ povm]

visualize(img) |> display
visualize(theo) |> display
##
visualize(stack([img, theo]))