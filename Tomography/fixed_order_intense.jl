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

x = axes(calibration, 1)
y = axes(calibration, 2)

fit_d, fit_c = calibration_fit(x, y, calibration)

fit_c.param
##
orders = 1:5

metrics = Matrix{Float32}(undef, length(orders), 100)

@showprogress for (m, order) ∈ enumerate(orders)
    images, ρs = load_data(path, order, (fit_d.param[5], fit_c.param[5]))

    basis_d = fixed_order_basis(order, fit_d.param)
    basis_c = fixed_order_basis(order, fit_c.param, -Float32(π) / 6)

    direct_povm = assemble_position_operators(x, y, basis_d)
    converted_povm = assemble_position_operators(x, y, basis_c)
    povm = stack((direct_povm, converted_povm))

    problem = StateTomographyProblem(povm)

    mthd = LinearInversion(problem)

    Threads.@threads for n ∈ axes(images, 4)
        probs = @view images[:, :, :, n]
        ρ = @view ρs[:, :, n]
        pred_ρ, _ = prediction(probs, mthd)

        metrics[m, n] = fidelity(ρ, pred_ρ)
    end
end

vec(mean(metrics, dims=2))
##
orders = 1:5
metrics_no_calib = Matrix{Float64}(undef, length(orders), 100)

p = Progress(prod(size(metrics_no_calib)))
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
        povm = stack((direct_povm, converted_povm))

        problem = StateTomographyProblem(povm)
        mthd = LinearInversion(problem)

        pred_ρ = prediction(slice, mthd)[1]
        metrics_no_calib[m, n] = fidelity(ρ, pred_ρ)
        next!(p)
    end
end
finish!(p)

vec(mean(metrics_no_calib, dims=2))

sort(metrics_no_calib, dims=2)[2,:]
##
h5open("Results/fixed_order_intense.h5", "cw") do file
    file["mean_fid"] = vec(mean(metrics, dims=2))
    file["std_fid"] = vec(std(metrics, dims=2))
    file["mean_fid_no_calib"] = vec(mean(metrics_no_calib, dims=2))
    file["std_fid_no_calib"] = vec(std(metrics_no_calib, dims=2))
    file["orders"] = collect(orders)
end