using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/obstructions.jl")
includet("../Utils/model_fitting.jl")
##
function load_data(path, key, bg)
    images, ρs, par = h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"], attrs(obj)["par"]
    end

    remove_background!(images, bg)
    images, ρs, par
end

path = "Data/blade.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

fit = calibration_fit(x, y, calibration)

bg = round(UInt8, fit.param[5])
basis = positive_l_basis(2, fit.param)
##
N = 3

fids = Matrix{Float64}(undef, 100, N)
pars = Vector{Float64}(undef, N)

for n ∈ 1:N
    images, ρs, par = load_data(path, "images_$n", bg)
    pars[n] = par[2]

    Is = get_valid_indices(x, y, blade_obstruction, x[Int(par[1])])
    povm = assemble_position_operators(x, y, basis)[Is]
    problem = StateTomographyProblem(povm)
    mthd = LinearInversion(problem)

    Threads.@threads for m ∈ axes(images, 3)
        probs = images[:, :, m][Is]
        ρ = ρs[:, :, m]
        pred_ρ = prediction(probs, mthd)[1]
        fids[m, n] = fidelity(ρ, pred_ρ)
    end
end

vec(mean(fids, dims=1))
##
pars
##
images, ρs, par = load_data(path, "images_2")
visualize(images[:,:,17])

##
h5open(saving_path, "cw") do file
    file["blade_pos"] = pars
end
##
path = "Data/Raw/blade_extra.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, 1, maximum(calibration), minimum(calibration)])
fit = surface_fit(gaussian_model, x, y, calibration, p0)
basis = positive_l_basis(2, fit.param)

images, ρs, par = load_data(path, "images_1")
par[2]

Is = get_valid_indices(x, y, blade_obstruction, x[Int(par[1])])
povm = assemble_position_operators(x, y, basis)[Is]
problem = StateTomographyProblem(povm)
mthd = LinearInversion(problem)

map!(x -> relu(x, round(UInt8, fit.param[6])), images, images)

metric = zero(Float64)

θs = Array{Float32}(undef, 3, 2, size(images)[end])
covs = Array{Float32}(undef, 3, 3, size(images)[end])

for m ∈ axes(images, 3)
    probs = images[:, :, m][Is]
    ρ = ρs[:, :, m]
    θ = gell_mann_projection(ρ)
    pred_ρ, pred_θ, cov = prediction(probs, mthd)

    θs[:, 1, m] = θ
    θs[:, 2, m] = pred_θ
    covs[:, :, m] .= cov

    metric += fidelity(ρ, pred_ρ) / size(images, 3)
end

metric
##

h5open(saving_path, "cw") do file
    file["thetas_4"] = θs
    file["covs_4"] = covs
end
##

h5open(saving_path, "cw") do file
    blade_pos = read(file["blade_pos"])[1:3]
    delete_object(file, "blade_pos")
    file["blade_pos"] = vcat(blade_pos, par[2])
end