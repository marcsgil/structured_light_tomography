using HDF5, QuantumMeasurements, ProgressMeter, LinearAlgebra

includet("../Utils/model_fitting.jl")
includet("../Utils/basis.jl")
includet("../Utils/bootstraping.jl")

function load_data(path, order, bgs)
    images, ρs = h5open(path) do file
        Float32.(read(file["images_order$order"])), read(file["labels_order$order"])
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
rs = Iterators.product(x, y)
npixels = length(x) * length(y)
sqrt_δA = sqrt((x[2] - x[1]) * (y[2] - y[1]) / 2) # We divide by 2 because we have two images

fit_d, fit_c = calibration_fit(x, y, calibration)

f!(buffer, r, pars) = fixed_order_basis!(buffer, r, pars)
g!(buffer, r, pars) = fixed_order_basis!(buffer, r, pars, Float32(π) / 6)


pars_d = (sqrt_δA, fit_d.param...)
pars_c = (sqrt_δA, fit_c.param...)
##
orders = 1:5

fid = Matrix{Float32}(undef, length(orders), 100)

@showprogress for (m, order) ∈ enumerate(orders)
    images, ρs = load_data(path, order, (fit_d.param[5], fit_c.param[5]))

    μ = empty_measurement(2 * npixels, order + 1, Matrix{Float32})
    μ1 = view(μ, 1:npixels, :)
    μ2 = view(μ, npixels+1:2*npixels, :)
    buffer = Matrix{ComplexF32}(undef, order + 1, 512)

    multithreaded_update_measurement!(μ1, buffer, rs, pars_d, f!)
    multithreaded_update_measurement!(μ2, buffer, rs, pars_c, g!)

    mthd = PreAllocatedLinearInversion(μ)

    Threads.@threads for n ∈ axes(images, 4)
        probs = @view images[:, :, :, n]
        ρ = @view ρs[:, :, n]
        pred_ρ = estimate_state(probs, μ, mthd)[1]

        fid[m, n] = fidelity(ρ, pred_ρ)
    end
end

vec(mean(fid, dims=2))
##
orders = 1:5
fid_no_calib = Matrix{Float64}(undef, length(orders), 100)

p = Progress(prod(size(fid_no_calib)))
Threads.@threads for m ∈ eachindex(orders)
    order = orders[m]
    images, ρs = load_data(path, order, (0x02, 0x02))

    for n ∈ axes(images, 4)
        slice = @view images[:, :, :, n]
        ρ = @view ρs[:, :, n]

        param_d = (sqrt_δA, center_of_mass_and_waist(view(slice, :, :, 1), order)...)
        param_c = (sqrt_δA, center_of_mass_and_waist(view(slice, :, :, 2), order)...)

        buffer = Matrix{ComplexF32}(undef, order + 1, 512)

        μ = empty_measurement(2 * npixels, order + 1, Matrix{Float32})
        μ1 = view(μ, 1:npixels, :)
        μ2 = view(μ, npixels+1:2*npixels, :)

        multithreaded_update_measurement!(μ1, buffer, rs, pars_d, f!)
        multithreaded_update_measurement!(μ2, buffer, rs, pars_c, g!)

        mthd = NormalEquations(μ)
        pred_ρ = estimate_state(slice, μ, mthd)[1]
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
##
# Single Image

path = "Data/fixed_order_intense.h5"

calibration = h5open(path) do file
    read(file["calibration"])
end

x = Float32.(axes(calibration, 1))
y = Float32.(axes(calibration, 2))
rs = Iterators.product(x, y)
npixels = length(x) * length(y)
sqrt_δA = sqrt((x[2] - x[1]) * (y[2] - y[1])) # We divide by 2 because we have two images

f!(buffer, r, pars) = fixed_order_basis!(buffer, r, pars)

orders = 1:5
fid = Matrix{Float64}(undef, length(orders), 100)

p = Progress(prod(size(fid)))
Threads.@threads for m ∈ eachindex(orders)
    order = orders[m]
    images, ρs = load_data(path, order, (0x02, 0x02))

    for n ∈ axes(images, 4)
        slice = @view images[:, :, 1, n]
        ρ = @view ρs[:, :, n]

        param = (sqrt_δA, center_of_mass_and_waist(slice, order)...)

        buffer = Matrix{ComplexF32}(undef, order + 1, 512)

        μ = empty_measurement(npixels, order + 1, Matrix{Float32})

        multithreaded_update_measurement!(μ, buffer, rs, param, f!)

        mthd = LinearInversion()
        pred_ρ = estimate_state(slice, μ, mthd)[1]
        fid[m, n] = fidelity(ρ, pred_ρ)
        next!(p)
    end
end
finish!(p)

vec(mean(fid, dims=2))
##
stack(bootstrap(slice) for slice ∈ eachslice(fid, dims=1))
##
h5open("Results/fixed_order_intense_single_image.h5", "cw") do file
    file["fid"] = stack(bootstrap(slice) for slice ∈ eachslice(fid, dims=1))
    file["orders"] = collect(orders)
end