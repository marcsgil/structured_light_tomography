using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/obstructions.jl")
includet("../Utils/model_fitting.jl")
includet("../Utils/bootstraping.jl")
##
function load_data(path, key, bg)
    images, ρs, par = h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"], attrs(obj)["par"]
    end

    remove_background!(images, bg)
    images, ρs, par
end

path = "Data/iris.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = axes(calibration, 1)
y = axes(calibration, 2)

fit = calibration_fit(x, y, calibration)

fit.param
bg = round(UInt8, fit.param[5])
basis = positive_l_basis(2, fit.param)
##
N = 3

fid = Matrix{Float64}(undef, 100, N)
pars = Vector{Float64}(undef, N)

for n ∈ 1:3
    images, ρs, par = load_data(path, "images_$n", bg)
    pars[n] = par[2]

    radius = (x[end] - x[begin]) * par[1]

    Is = get_valid_indices(x, y, iris_obstruction, fit.param[1], fit.param[2], radius)
    povm = assemble_position_operators(x, y, basis)[Is]
    problem = StateTomographyProblem(povm)
    mthd = LinearInversion(problem)

    for m ∈ axes(images, 3)
        probs = images[:, :, m][Is]
        ρ = ρs[:, :, m]
        pred_ρ = prediction(probs, mthd)[1]

        fid[m, n] = fidelity(ρ, pred_ρ)
    end
end

mean(fid, dims=1)
##
statistics = stack(bootstrap(slice) for slice ∈ eachslice(fid, dims=2))
(statistics[3, :] - statistics[2, :]) / 2
##
h5open("Results/iris.h5", "cw") do file
    file["fid"] = stack(bootstrap(slice) for slice ∈ eachslice(fid, dims=2))
    file["radius"] = pars
end