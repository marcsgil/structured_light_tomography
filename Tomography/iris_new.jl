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

path = "Data/Raw/test_python_4.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

function simple_model(x, y, p)
    x₀, y₀, w, α, A, bg = p
    bg + A * exp.(-2 * ((x .- x₀) .^ 2 .+ α *(y .- y₀) .^ 2) ./ w^2)
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, 1, maximum(calibration), minimum(calibration)])
fit = surface_fit(simple_model, x, y, calibration, p0)


basis = positive_l_basis(2, fit.param)
##
N = 1

fids = Matrix{Float64}(undef, 50, N)
pars = Vector{Float64}(undef, N)

for n ∈ 1:N
    images, ρs = load_data(path, "images_$n")

    #pars[n] = par[2]

    obstructed_basis = get_obstructed_basis(basis, iris_obstruction, 0, 0,0.5)
    povm, ρs = get_proper_povm_and_states(x, y, ρs, obstructed_basis)

    map!(x -> relu(x, 0x04), images, images)
    #povm = assemble_position_operators(x, y, basis)
    mthd = LinearInversion(povm)

    for m ∈ axes(images, 3)
        σ, _ = prediction(Float32.(images[:, :, m]), mthd)
        fids[m, n] = fidelity(conj.(ρs[:, :, m]), σ)
    end
end

fids

argmin(fids)
mean(fids, dims=1)
##
images, ρs = load_data(path, "images_1")

calibration = h5open(path) do file
    file["calibration"] |> read
end

"""x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, 1, maximum(calibration), minimum(calibration)])

fit = surface_fit(gaussian_model, x, y, calibration[:,:,1], p0)"""

basis = positive_l_basis(2, new_fit_param)
##
visualize(calibration[:, :, 1]) |> display

visualize([gaussian_model(x, y, fit.param) for x ∈ x, y ∈ y]) |> display
##
n = 1
visualize(images[:, :, n]) |> display
visualize(get_intensity(conj.(ρs[:, :, n]), basis, x, y)) |> display
##
visualize(abs2.(lg(x .- new_fit_param[1], y .- new_fit_param[2], w=fit.param[3], l=2)))
##


test = h5open("Data/test_side.h5") do file
    read(file["test"])
end

visualize(test[:, :, 2]) |> display
##

fit.param[3]
new_fit_param[3]

basis = positive_l_basis(2, new_fit_param)
se(x, y) = mapreduce((x, y) -> abs2(x - y), +, x, y)

pred = [gaussian_model(x, y, new_fit_param) for x ∈ x, y ∈ y]
se(pred, calibration)
##
most_frequent_value(data) = countmap(data) |> argmax

function remove_background!(images)
    bg = most_frequent_value(images)
    map!(x -> x < bg ? zero(x) : x - bg, images, images)
end