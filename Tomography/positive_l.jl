using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra, FiniteDiff
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
        read(obj), attrs(obj)["density_matrices"]
    end
end

path = "Data/Raw/positive_l.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, 1, maximum(calibration), minimum(calibration)])
fit = surface_fit(gaussian_model, x, y, calibration, p0)

fit.param
##
dims = 2:6

fids = Matrix{Float64}(undef, 100, length(dims))
errors = Matrix{Float64}(undef, 100, length(dims))
pars = Vector{Float64}(undef, length(dims))

for n ∈ eachindex(dims)
    images, ρs = load_data(path, "images_dim$(dims[n])")

    basis = positive_l_basis(dims[n], fit.param)
    povm = assemble_position_operators(x, y, basis)
    problem = StateTomographyProblem(povm)
    mthd = LinearInversion(problem)

    map!(x -> relu(x, round(UInt8, fit.param[6])), images, images)

    for m ∈ axes(images, 3)
        probs = normalize(images[:, :, m], 1)
        ρ = ρs[:, :, m]
        θ = gell_mann_projection(ρ)
        ρ_pred, θ_pred, cov = prediction(probs, mthd)
        fids[m, n] = fidelity(ρ, θ_pred)
        #grad = FiniteDiff.finite_difference_gradient(θ -> fidelity(ρ, θ), θ_pred)
        #errors[m, n] = dot(grad, cov, grad) * 1.96
        fids[m, n] = sum(abs2, θ_pred - θ)
        errors[m, n] = sqrt(2 * cov ⋅ cov) * 1.96
    end
end

fids

mean(fids, dims=1)

mean(errors, dims=1)
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