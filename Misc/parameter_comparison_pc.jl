using TiffImages
include("../Utils/model_fitting.jl")

calibration = reshape(real.(TiffImages.load("Data/Raw/Photocount/calibration.tif")), (64, 64, 2))
reverse!(calibration, dims=3) #So that the direct image comes first
calibration = reinterpret(UInt16, calibration)

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, maximum(calibration), minimum(calibration)])

fit_d = surface_fit(gaussian_model, x, y, calibration[:, :, 1], p0)
fit_c = surface_fit(gaussian_model, x, y, calibration[:, :, 2], p0)

visualize(calibration)

calibration_pred = [gaussian_model(x, y, fit.param) for x ∈ x, y ∈ y, fit ∈ (fit_d, fit_c)]
visualize(calibration_pred)