using HDF5, BayesianTomography, ProgressMeter

includet("../Utils/model_fitting.jl")
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/bootstraping.jl")

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
order = 1

images, ρs = load_data(path, order, (fit_d.param[5], fit_c.param[5]))


basis_d = fixed_order_basis(order, fit_d.param)
basis_c = fixed_order_basis(order, fit_c.param, -Float32(π) / 6)


assemble_position_operators(x, y, basis_d)

sum(direct_povm)

converted_povm = assemble_position_operators(x, y, basis_c)
povm = stack((direct_povm, converted_povm))

@benchmark StateTomographyProblem($povm)

problem = StateTomographyProblem(povm)

mthd = LinearInversion(problem)

function get_real_representation(measurement)
    T = real(eltype(eltype(measurement)))
    dim = size(first(measurement), 1)
    traceless_part = Matrix{T}(undef, length(measurement), dim^2 - 1)
    trace_part = Vector{T}(undef, length(measurement))

    Threads.@threads for n ∈ eachindex(measurement)
        trace_part[n] = real(tr(measurement[n])) / dim
        gell_mann_projection!(view(traceless_part, n, :), measurement[n])
    end

    traceless_part, trace_part
end

get_real_representation(povm)

@benchmark get_real_representation($povm)