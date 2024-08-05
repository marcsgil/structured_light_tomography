using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/obstructed_measurements.jl")

relu(x, y) = x > y ? x - y : zero(x)

function load_data(path, key)
    h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"], attrs(obj)["par"]
    end
end

path = "Data/Raw/Old/iris.h5"

fit_param, x, y = h5open(path) do file
    obj = file["fit_param"]
    read(obj), attrs(obj)["x"], attrs(obj)["y"]
end

basis = positive_l_basis(2, fit_param[1:4])
##
N = 5

fids = Matrix{Float64}(undef, 100, N)
pars = Vector{Float64}(undef, N)

for n ∈ 1:N
    images, ρs, par = load_data(path, "images_$n")

    pars[n] = par[2]

    map!(x -> relu(x, 0x02), images, images)

    obstructed_basis = get_obstructed_basis(basis, iris_obstruction, fit_param[1], fit_param[2], par[1])
    povm, ρs = get_proper_povm_and_states(x, y, ρs, obstructed_basis)

    mthd = LinearInversion(povm)

    for m ∈ axes(images, 3)
        σ, _ = prediction(Float32.(images[:, :, m]), mthd)
        fids[m, n] = fidelity(ρs[:, :, m], σ)
    end
end

mean(fids, dims=1)
##
I = sortperm(pars)
pars[I]

mean(fids, dims=1)[I]