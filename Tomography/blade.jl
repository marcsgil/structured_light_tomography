using BayesianTomography, HDF5, PositionMeasurements, ProgressMeter, LinearAlgebra
includet("../Utils/basis.jl")

fit_param, x, y = h5open("Data/Raw/blade.h5") do file
    obj = file["fit_param"]
    read(obj), attrs(obj)["x"], attrs(obj)["y"]
end

fit_param[3] = fit_param[3]

images, ρs, par = h5open("Data/Raw/blade.h5") do file
    obj = file["images_1"]

    read(obj), attrs(obj)["density_matrices"], attrs(obj)["par"]
end

relu(x, y) = x > y ? x - y : zero(x)

treated_images = [relu(x, 0x03) for x ∈ images]

_basis = positive_l_basis(2, fit_param[1:4])
basis = [(x, y) -> f(x, y) * (x < par[1]) for f ∈ _basis]

povm = assemble_position_operators(x, y, basis)

mthd = LinearInversion(povm)

fids = Vector{Float64}(undef, size(treated_images, 3))

for n ∈ axes(treated_images, 3)
    σ, _ = prediction(Float32.(images[:, :, n]), mthd)
    fids[n] = fidelity(ρs[:, :, n], σ)
end

fids
mean(fids)
##
n = 4
visualize(images[:, :, n])

visualize([real(tr(ρs[:, :, n] * Π)) for Π ∈ povm])

h5open("Data/Raw/blade.h5") do file
    read(file["calibration"])
end |> visualize