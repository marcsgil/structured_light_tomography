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

fit_c.param
##
orders = 1:5

fid = Matrix{Float32}(undef, length(orders), 100)

@showprogress for (m, order) ∈ enumerate(orders)
    images, ρs = load_data(path, order, (fit_d.param[5], fit_c.param[5]))

    basis_d = fixed_order_basis(order, fit_d.param)
    basis_c = fixed_order_basis(order, fit_c.param, -Float32(π) / 6)

    measurement = Measurement(assemble_position_operators(x, y, basis_d, basis_c))

    mthd = PreAllocatedLinearInversion(measurement)

    Threads.@threads for n ∈ axes(images, 4)
        probs = @view images[:, :, :, n]
        ρ = @view ρs[:, :, n]
        pred_ρ, _ = prediction(probs, measurement, mthd)

        fid[m, n] = fidelity(ρ, pred_ρ)
    end
end

vec(mean(fid, dims=2))
##
orders = 1:5
fid_no_calib = Matrix{Float64}(undef, length(orders), 100)
mthd = LinearInversion()

p = Progress(prod(size(fid_no_calib)))
Threads.@threads for m ∈ eachindex(orders)
    order = orders[m]
    images, ρs = load_data(path, order, (0x02, 0x02))

    for n ∈ axes(images, 4)
        slice = @view images[:, :, :, n]
        ρ = @view ρs[:, :, n]

        param_d = center_of_mass_and_waist(view(slice, :, :, 1), order)
        param_c = center_of_mass_and_waist(view(slice, :, :, 2), order)

        basis_d = fixed_order_basis(order, param_d)
        basis_c = fixed_order_basis(order, param_c, -Float32(π) / 6)

        direct_povm = assemble_position_operators(x, y, basis_d)
        converted_povm = assemble_position_operators(x, y, basis_c)
        measurement = Measurement(stack((direct_povm, converted_povm)))

        pred_ρ = prediction(slice, measurement, mthd)[1]
        fid_no_calib[m, n] = fidelity(ρ, pred_ρ)
        next!(p)
    end
end
finish!(p)

vec(mean(fid_no_calib, dims=2))
##
h5open("Results/fixed_order_intense.h5", "cw") do file
    file["fid"] = stack(bootstrap(slice) for slice ∈ eachslice(fid, dims=1))
    file["fid_no_calib"] = stack(bootstrap(slice) for slice ∈ eachslice(fid_no_calib, dims=1))
    file["orders"] = collect(orders)
end