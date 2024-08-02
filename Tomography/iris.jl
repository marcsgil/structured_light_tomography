using BayesianTomography, HDF5, PositionMeasurements, ProgressMeter, LinearAlgebra
includet("../Utils/basis.jl")

path = "Data/Raw/iris.h5"

fit_param, x, y = h5open(path) do file
    obj = file["fit_param"]
    read(obj), attrs(obj)["x"], attrs(obj)["y"]
end

images, ρs, par = h5open(path) do file
    obj = file["images_1"]

    read(obj), attrs(obj)["density_matrices"], attrs(obj)["par"]
end

fit_param[3] = fit_param[3] / √2

relu(x, y) = x > y ? x - y : zero(x)

treated_images = [relu(x, 0x02) for x ∈ images]

_basis = positive_l_basis(2, fit_param[1:4])
x₀ = fit_param[1]
y₀ = fit_param[2]
basis = [(x, y) -> f(x, y) * ((x - x₀)^2 + (y - y₀)^2 < par[1]^2) for f ∈ _basis]

povm = assemble_position_operators(x, y, basis)
##
mthd = LinearInversion(povm)

fids = Vector{Float64}(undef, size(treated_images, 3))

for n ∈ axes(treated_images, 3)
    σ, _ = prediction(Float32.(treated_images[:, :, n]), mthd)
    fids[n] = fidelity(ρs[:, :, n], σ)
end

mean(fids)
##
using CairoMakie
visualize(treated_images[:, :, 2])
ρ = ρs[:, :, 1]
σ = prediction(images[:, :, 1], mthd)[1]

visualize([real(tr(ρ * Π)) for Π ∈ povm])

sum(Int, images[:, :, 1])
##