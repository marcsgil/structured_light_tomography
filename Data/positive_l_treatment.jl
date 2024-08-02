using HDF5, LinearAlgebra

include("../Utils/model_fitting.jl")

file = h5open("Data/Raw/positive_l.h5")
out = h5open("Data/Processed/positive_l.h5", "cw")

calibration = read(file["calibration"])

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, 1, maximum(calibration), minimum(calibration)])
fit = surface_fit(gaussian_model, x, y, calibration, p0)

fit.param

out["fit_param"] = fit.param

for dim ∈ 2:6
    images = Float64.(read(file["images_dim$dim"]))

    images .-= round(UInt8, fit.param[6])

    out["images_dim$dim"] = images
    out["labels_dim$dim"] = read(file["labels_dim$dim"])
end

close(out)
close(file)