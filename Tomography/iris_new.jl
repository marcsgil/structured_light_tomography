using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/obstructed_measurements.jl")
includet("../Utils/model_fitting.jl")

relu(x, y) = x > y ? x - y : zero(x)
extract_θ(ρ, ωs) = stack(real(tr(ρ * ω)) for ω ∈ eachslice(view(ωs, :, :, 2:4), dims=3))
ωs = gell_mann_matrices(2)

function load_data(path, key)
    h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"], attrs(obj)["par"]
    end
end

path = "Data/Raw/test_julia.h5"

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

    basis_func_obs = get_obstructed_basis(basis, iris_obstruction, fit.param[1], fit.param[2], par[1])
    T, Ω, L = assemble_povm_matrix(x, y, basis_func_obs)
    mthd = LinearInversion(T, Ω)

    map!(x -> relu(x, 0x03), images, images)

    for m ∈ axes(images, 3)
        probs = vec(normalize(images[:, :, m], 1))
        σ, _ = prediction(probs, mthd)
        fids[m, n] = fidelity(ρs[:, :, m], σ / tr(σ))
    end
end

fids

mean(fids, dims=1)
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