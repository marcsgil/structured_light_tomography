using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/incomplete_measurements.jl")
includet("../Utils/model_fitting.jl")
##
relu(x, y) = x > y ? x - y : zero(x)

function load_data(path, key)
    h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"], attrs(obj)["par"]
    end
end

path = "Data/Raw/iris_new.h5"
saving_path = "Results/Intense/iris.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, 1, maximum(calibration), minimum(calibration)])
fit = surface_fit(gaussian_model, x, y, calibration, p0)

fit.param
basis = positive_l_basis(2, fit.param)
##
N = 3

metrics = Matrix{Float64}(undef, 100, N)
errors = Matrix{Float64}(undef, 100, N)

for n ∈ 1:N
    images, ρs, par = load_data(path, "images_$n")
    pars[n] = par[2]

    Is = get_valid_indices(x, y, iris_obstruction, fit.param[1], fit.param[2], par[1])
    povm = assemble_position_operators(x, y, basis)[Is]
    L = transform_incomplete_povm!(povm)
    problem = StateTomographyProblem(povm)
    mthd = LinearInversion(problem)

    map!(x -> relu(x, round(UInt8, fit.param[6])), images, images)

    θs = Array{Float32}(undef, 3, 2, size(images)[end])
    covs = Array{Float32}(undef, 3, 3, size(images)[end])

    Threads.@threads for m ∈ axes(images, 3)
        probs = images[:, :, m][Is]
        ρ = ρs[:, :, m]
        θ = gell_mann_projection(ρ)
        pred_ρ, pred_θ, cov = prediction(probs, mthd)
        density_matrix_transform!(pred_ρ, inv(L))
        gell_mann_projection!(pred_θ, pred_ρ)

        θs[:, 1, m] = θ
        θs[:, 2, m] = pred_θ
        covs[:, :, m] = cov
    end

    h5open(saving_path, "cw") do file
        file["thetas_$n"] = θs
        file["covs_$n"] = covs
    end
end

h5open(saving_path, "cw") do file
    file["radius"] = pars
end
##
fids

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