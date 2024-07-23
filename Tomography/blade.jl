using BayesianTomography, HDF5, PositionMeasurements, ProgressMeter, LinearAlgebra
includet("../Utils/basis.jl")

fit_param, x, y = h5open("Data/Raw/blade.h5") do file
    obj = file["fit_param"]
    read(obj), attrs(obj)["x"], attrs(obj)["y"]
end

images, ρs, par = h5open("Data/Raw/blade.h5") do file
    obj = file["images_5"]

    read(obj), attrs(obj)["density_matrices"], attrs(obj)["par"]
end

relu(x, y) = x > y ? x - y : zero(x)

treated_images = [relu(x, 0x02) for x ∈ images]

_basis = positive_l_basis(2, fit_param[1:4])
basis = [(x, y) -> f(x, y) * (x < par[1]) for f ∈ _basis]

povm = assemble_position_operators(x, y, basis)

mthd = LinearInversion(povm)

fids = Vector{Float64}(undef, size(treated_images, 3))

for n ∈ axes(treated_images, 3)
    σ = prediction(Float32.(treated_images[:, :, n]), mthd)
    fids[n] = fidelity(ρs[:, :, n], σ)
end

mean(fids)
##
visualize(treated_images[:, :, 8])
ρs[:, :, 1]
prediction(images[:, :, 1], mthd)

sum(Int, images[:, :, 1])
##