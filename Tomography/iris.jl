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

fids = Matrix{Float64}(undef, 100, N)
pars = Vector{Float64}(undef, N)

for n ∈ 1:N
    images, ρs, par = load_data(path, "images_$n")
    pars[n] = par[2]

    Is = get_valid_indices(x, y, iris_obstruction, fit.param[1], fit.param[2], par[1])
    povm = assemble_position_operators(x, y, basis)[Is]
    L = transform_incomplete_povm!(povm)
    problem = StateTomographyProblem(povm)
    mthd = LinearInversion(problem)

    map!(x -> relu(x, round(UInt8, fit.param[6])), images, images)

    for m ∈ axes(images, 3)
        probs = images[:, :, m][Is]
        σ, _ = prediction(probs, mthd)
        σ = project2density(σ)
        density_matrix_transform!(σ, inv(L))
        #fids[m, n] = fidelity(ρs[:, :, m], σ)
        fids[m, n] = real(tr((ρs[:, :, m] - σ)^2))
    end
end

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
m=6
θ = extract_θ(ρs[:, :, m], ωs)
η = η_func(θ, ωs, L, ωs)

visualize(reshape(get_probs(mthd, η), 200, 200))
visualize(images[:,:,m])
##