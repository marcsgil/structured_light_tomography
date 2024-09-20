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

path = "Data/iris.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

fit = calibration_fit(x, y, calibration)

fit.param
bg = round(UInt8, fit.param[5])
basis = positive_l_basis(2, fit.param)
##



##
N = 3

fids = Matrix{Float64}(undef, 100, N)
pars = Vector{Float64}(undef, N)

for n ∈ 1:3
    images, ρs, par = load_data(path, "images_$n", bg)
    pars[n] = par[2]

    Is = get_valid_indices(x, y, iris_obstruction, fit.param[1], fit.param[2], par[1])
    povm = assemble_position_operators(x, y, basis)[Is]
    problem = StateTomographyProblem(povm)
    mthd = LinearInversion(problem)

    for m ∈ axes(images, 3)
        probs = images[:, :, m][Is]
        ρ = ρs[:, :, m]
        pred_ρ = prediction(probs, mthd)[1]

        fids[m, n] = fidelity(ρ, pred_ρ)

        if m==5 && n ==3
            display(ρ)
            display(pred_ρ)
        end
    end
end

mean(fids, dims=1)
##
std(fids, dims=1)
pars
##
h5open(saving_path, "cw") do file
    file["radius"] = pars
end
##

mean(fids, dims=1)
pars
##
images, ρs, par = load_data(path, "images_2")

calibration = h5open(path) do file
    file["calibration"] |> read
end

basis_func_obs = get_obstructed_basis(basis, iris_obstruction, fit.param[1], fit.param[2], par[1])
T, Ω, L = assemble_povm_matrix(x, y, basis_func_obs)
mthd = LinearInversion(T, Ω)
##
m = 6
θ = extract_θ(ρs[:, :, m], ωs)
η = η_func(θ, ωs, L, ωs)

visualize(reshape(get_probs(mthd, η), 200, 200))
visualize(images[:, :, m])
##