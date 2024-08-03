using BayesianTomography, HDF5, PositionMeasurements, ProgressMeter, LinearAlgebra
includet("../Utils/basis.jl")
includet("../Utils/model_fitting.jl")

file = h5open("Data/Raw/test.h5")
#fit_param = read(file["fit_param"])

calibration = read(file["calibration"])

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, 1, maximum(calibration), minimum(calibration)])
fit = surface_fit(gaussian_model, x, y, calibration, p0)

fit_param = fit.param
##
fids = Vector{Float64}(undef, 5)

basis = positive_l_basis(2, fit_param[1:4])
povm = assemble_position_operators(x, y, basis)

mthd = LinearInversion(povm)

images = file["images_1"][:, :, 1:5]
images .-= round(UInt8, 1)
#ρs = file["labels_dim2"][:, :, 1:5]
ρs = attrs(file["images_1"])["density_matrices"]

for (n, probs) ∈ enumerate(eachslice(images, dims=3))
    σ, _ = prediction(probs, mthd)
    fids[n] = fidelity(ρs[:, :, n], σ)
end

fids
mean(fids)
##
using CairoMakie
visualize(images[:,:,1])