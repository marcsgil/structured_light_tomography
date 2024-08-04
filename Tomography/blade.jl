using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")

relu(x, y) = x > y ? x - y : zero(x)

function load_data(path, key)
    h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"], attrs(obj)["par"]
    end
end

function get_povm(x, y, cutoff, fit_param)
    basis = [(x, y) -> f(x, y) * (x < cutoff) for f ∈ positive_l_basis(2, fit_param[1:4])]
    assemble_position_operators(x, y, basis)
end

path = "Data/Raw/Old/blade.h5"

fit_param, x, y = h5open(path) do file
    obj = file["fit_param"]
    read(obj), attrs(obj)["x"], attrs(obj)["y"]
end
##
N = 5

fids = Matrix{Float64}(undef, 100, N)

for n ∈ 1:N
    images, ρs, par = load_data(path, "images_$n")

    map!(x -> relu(x, 0x02), images, images)

    povm = get_povm(x, y, x[Int(par[1])], fit_param)
    mthd = LinearInversion(povm)

    for m ∈ axes(images, 3)
        σ, _ = prediction(Float32.(images[:, :, m]), mthd)
        fids[m, n] = fidelity(ρs[:, :, m], σ)
    end
end

mean(fids, dims=1)