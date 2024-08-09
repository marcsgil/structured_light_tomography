using HDF5, StructuredLight

includet("../Utils/obstructed_measurements.jl")
includet("../Utils/model_fitting.jl")
includet("../Experiment/AcquisitionUtils/capture_func.jl")
##
calibration_fund = h5open("Data/Raw/test_python_4.h5") do file
    file["calibration_fund"] |> read
end

x = LinRange(-0.5, 0.5, size(calibration_fund, 1))
y = LinRange(-0.5, 0.5, size(calibration_fund, 2))

p0 = Float64.([0, 0, 0.1, 1, maximum(calibration_fund), minimum(calibration_fund)])
fit = surface_fit(gaussian_model, x, y, calibration_fund, p0)

fit.param
##
calibration = h5open("Data/Raw/test_python_4.h5") do file
    file["calibration"] |> read
end

r = find_iris_radius(calibration, x, y, fit.param[1], fit.param[2])
r / fit.param[3]
##

using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/obstructed_measurements.jl")
includet("../Utils/model_fitting.jl")

relu(x, y) = x > y ? x - y : zero(x)

function load_data(path, key)
    h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"]
    end
end

path = "Data/Raw/test_python_4.h5"

basis = positive_l_basis(2, fit.param)
##
N = 1

fids = Matrix{Float64}(undef, 50, N)
pars = Vector{Float64}(undef, N)

for n ∈ 1:N
    images, ρs = load_data(path, "images_$n")

    #pars[n] = par[2]

    obstructed_basis = get_obstructed_basis(basis, iris_obstruction, fit.param[1], fit.param[2], 0.9*r)
    povm, new_ρs = get_proper_povm_and_states(x, y, ρs, obstructed_basis)

    map!(x -> relu(x, 0x02), images, images)
    #povm = assemble_position_operators(x, y, basis)
    mthd = LinearInversion(povm)

    for m ∈ axes(images, 3)
        σ, _ = prediction(Float32.(images[:, :, m]), mthd)
        fids[m, n] = fidelity(conj.(new_ρs[:, :, m]), σ)
    end
end

mean(fids)
##
images, ρs = load_data(path, "images_1")

obstructed_basis = get_obstructed_basis(basis, iris_obstruction, fit.param[1], fit.param[2], 0.9r)
povm, new_ρs = get_proper_povm_and_states(x, y, ρs, obstructed_basis)
n = 1
visualize(images[:, :, n]) |> display
visualize(get_intensity(conj.(ρs[:, :, n]), obstructed_basis, x, y)) |> display
visualize([real(tr(conj.(new_ρs[:,:,n])*Π)) for Π in povm])
##