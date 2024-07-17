using BayesianTomography, HDF5, LsqFit, PositionMeasurements, ProgressMeter
includet("../Utils/model_fitting.jl")
includet("../Utils/basis.jl")

file = h5open("Data/Processed/positive_l.h5")
calibration = read(file["calibration"])

p0 = Float64.([maximum(calibration), 0, 0, 0.1, 1, 3])

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

xy = hcat(([x, y] for x in x, y in y)...)

twoD_Gaussian(xy, p0)

fit = LsqFit.curve_fit(twoD_Gaussian, xy, calibration |> vec, p0)
@assert fit.converged
fit.param
##
margin_of_error = margin_error(fit, 0.05)

margin_of_error ./ fit.param
confidence_inter = confint(fit; level=0.95)
##
dim = 6
basis = positive_l_basis(dim, fit.param[2], fit.param[3], fit.param[4], fit.param[5])
povm = assemble_position_operators(x, y, basis)

mthd = LinearInversion(povm)

images = read(file["images_dim$dim"]) .- round(UInt8, fit.param[6])
ρs = read(file["labels_dim$dim"])

fids = Vector{Float64}(undef, size(images, 3))

for (n, probs) ∈ enumerate(eachslice(images, dims=3))
    σ = prediction(probs, mthd)
    fids[n] = fidelity(ρs[:, :, n], σ)
end

mean(fids)
##
dims = 2:3
fids = Matrix{Float64}(undef, length(dims), 100)

@showprogress for (m, dim) ∈ enumerate(dims)
    images = read(file["images_dim$dim"])
    ρs = read(file["labels_dim$dim"])
    basis = positive_l_basis(dim, fit.param[2], fit.param[3], fit.param[4], fit.param[5])

    povm = assemble_position_operators(x, y, basis)
    mthd = LinearInversion(povm)

    for (n, probs) ∈ enumerate(eachslice(images, dims=3))
        σ = prediction(probs, mthd)
        fids[m, n] = fidelity(ρs[:, :, n], σ)
    end
end

dropdims(mean(fids, dims=2), dims=2)
##