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

path = "Data/blade.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = axes(calibration, 1)
y = axes(calibration, 2)

fit = calibration_fit(x, y, calibration)

bg = round(UInt8, fit.param[5])
basis = positive_l_basis(2, fit.param)
##
N = 3

fid = Matrix{Float64}(undef, 100, N)
pars = Vector{Float64}(undef, N)

for n ∈ 1:N
    images, ρs, par = load_data(path, "images_$n", bg)
    pars[n] = par[2]

    Is = get_valid_indices(x, y, blade_obstruction, par[1])
    povm = assemble_position_operators(x, y, basis)[Is]
    problem = StateTomographyProblem(povm)
    mthd = LinearInversion(problem)

    Threads.@threads for m ∈ axes(images, 3)
        probs = images[:, :, m][Is]
        ρ = ρs[:, :, m]
        pred_ρ = prediction(probs, mthd)[1]
        fid[m, n] = fidelity(ρ, pred_ρ)
    end
end

vec(mean(fid, dims=1))
##
stack(bootstrap(slice) for slice ∈ eachslice(fid, dims=2))
##
h5open("Results/blade.h5", "cw") do file
    file["fid"] = stack(bootstrap(slice) for slice ∈ eachslice(fid, dims=2))
    file["blade_pos"] = pars
end
