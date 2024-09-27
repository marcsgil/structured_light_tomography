using HDF5, BayesianTomography, ProgressMeter

includet("../Utils/model_fitting.jl")
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")

function load_data(path, order, bgs)
    images, ρs = h5open(path) do file
        Float32.(read(file["images_order$order"])), conj.(read(file["labels_order$order"]))
    end

    for (n, slice) ∈ enumerate(eachslice(images, dims=3))
        remove_background!(slice, bgs[n])
    end

    for slice ∈ eachslice(images, dims=(3, 4))
        normalize!(slice, 1)
    end
    images ./= 2

    images, ρs
end
##
path = "Data/fixed_order_intense.h5"

calibration = h5open(path) do file
    read(file["calibration"])
end

x = Float32.(axes(calibration, 1))
y = Float32.(axes(calibration, 2))

fit_d, fit_c = calibration_fit(x, y, calibration)

fit_d.param
##
order = 5

images, ρs = load_data(path, order, (fit_d.param[5], fit_c.param[5]))


images

basis_d = fixed_order_basis(order, fit_d.param)
basis_c = fixed_order_basis(order, fit_c.param, -Float32(π) / 6)
assemble_position_operators(x, y, basis_d, basis_c)

Measurement(assemble_position_operators(x, y, basis_d, basis_c))
##

@benchmark Measurement(assemble_position_operators($x, $y, $basis_d, $basis_c))

@benchmark assemble_position_operators($x, $y, 6, $basis_d, $basis_c)
##
ψ = BayesianTomography.sample(HaarVector(6))
θ = Vector{Float32}(undef, 6^2 - 1)
ρ = ψ * ψ'

@benchmark gell_mann_projection!($θ, $ψ)
@benchmark gell_mann_projection!($θ, $ρ)
##

mthd = PreAllocatedLinearInversion(m)

@benchmark PreAllocatedLinearInversion($m)

img = calibration[:, :, 1]
@benchmark center_of_mass_and_waist($img, 0)

@code_warntype prediction(calibration, m, mthd)

@benchmark prediction($calibration, $m, $mthd)
##
A = rand(Float32, 400^2, 35)
q = rand(Float32, 400^2)
A \ q

@benchmark $A \ $q