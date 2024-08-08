using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/obstructed_measurements.jl")
includet("../Utils/model_fitting.jl")

relu(x, y) = x > y ? x - y : zero(x)

function load_data(path, key)
    h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"]
    end
end

path = "Data/Raw/test_simple.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, 1, maximum(calibration), minimum(calibration)])
fit = surface_fit(gaussian_model, x, y, calibration, p0)

fit_param = fit.param
fit_param[3] /= 1.2

basis = positive_l_basis(2, fit_param[1:4])
##
N = 1

fids = Matrix{Float64}(undef, 3, N)
pars = Vector{Float64}(undef, N)

for n ∈ 1:N
    images, ρs = load_data(path, "image_$n")

    pars[n] = par[2]

    map!(x -> relu(x, 0x04), images, images)
    povm = assemble_position_operators(x, y, basis)
    mthd = LinearInversion(povm)

    for m ∈ axes(images, 3)
        σ, _ = prediction(Float32.(images[:, :, m]), mthd)
        fids[m, n] = fidelity(ρs[:, :, m], σ)
    end
end

fids

mean(fids, dims=1)
##
images, ρs = h5open("Data/Raw/mixed_intense.h5") do file
    read(file["images_order1"]), read(file["labels_order1"])
end

calibration = h5open("Data/Raw/mixed_intense.h5") do file
    file["calibration"] |> read
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, 1, maximum(calibration), minimum(calibration)])

"""x = axes(calibration,1)
y = axes(calibration, 2)
x₀ = size(calibration, 1) ÷ 2
y₀ = size(calibration, 2) ÷ 2

w = x₀ ÷ 4

p0 = [x₀, y₀, w, 1., maximum(calibration), minimum(calibration)]"""

fit = surface_fit(gaussian_model, x, y, calibration[:,:,1], p0)

basis = fixed_order_basis(1, fit.param) |> reverse
##
visualize(calibration[:,:,1]) |> display

visualize([gaussian_model(x, y, fit.param) for x ∈ x, y ∈ y]) |> display
##
n = 11
visualize(images[:, :, 1, n]) |> display
visualize(get_intensity(ρs[:, :, n], basis, x, y))  |> display
##
visualize(abs2.(lg(x.-new_fit_param[1], y .- new_fit_param[2], w=fit.param[3], l=2)))
