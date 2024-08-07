using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/obstructed_measurements.jl")
includet("../Utils/model_fitting.jl")

ρs = sample(ProductMeasure(2), 100)

rs = Base.oneto(200)
x₀ = rs[length(rs)÷2]
y₀ = rs[length(rs)÷2]
w = length(rs)÷8

p0 = [x₀, y₀, w, 1, 1, 2.0]
basis = positive_l_basis(2, p0)

calibration = [gaussian_model(x, y, p0) for x ∈ rs, y ∈ rs]

images = stack(get_intensity(ρ, basis, rs, rs) for ρ ∈ eachslice(ρs, dims=3))

visualize(calibration)
visualize(images[:, :, 1:2])

fit = surface_fit(gaussian_model, rs, rs, calibration, p0)

fit_param = fit.param

##
fids = Vector{Float64}(undef, size(ρs, 3))
##
povm = assemble_position_operators(rs, rs, basis)
mthd = LinearInversion(povm)

for m ∈ axes(images, 3)
    σ, _ = prediction(images[:, :, m], mthd)
    fids[m] = fidelity(ρs[:, :, m], σ)
end

maximum(fids)

mean(fids)