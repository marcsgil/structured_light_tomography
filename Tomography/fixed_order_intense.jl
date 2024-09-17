using HDF5, BayesianTomography, ProgressMeter

includet("../Utils/model_fitting.jl")
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")

relu(x::T1, y::T2) where {T1,T2} = max(zero(promote_type(T1, T2)), x - y)

function load_data(path, order, bgs)
    images, ρs = h5open(path) do file
        Float32.(read(file["images_order$order"])), conj.(read(file["labels_order$order"]))
    end

    Threads.@threads for J ∈ eachindex(IndexCartesian(), images)
        images[J] = relu(images[J], bgs[J[3]])
    end

    for slice ∈ eachslice(images, dims=(3, 4))
        normalize!(slice, 1)
    end
    images ./= 2

    images, ρs
end
##
path = "Data/Raw/fixed_order_intense.h5"

calibration = h5open(path) do file
    read(file["calibration"])
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, maximum(calibration), minimum(calibration)])

fit_d = surface_fit(gaussian_model, x, y, calibration[:, :, 1], p0)
fit_c = surface_fit(gaussian_model, x, y, calibration[:, :, 2], p0)

fit_d.param
##
orders = 1:5

metrics = Matrix{Float64}(undef, length(orders), 100)

@showprogress for (m, order) ∈ enumerate(orders)
    images, ρs = load_data(path, order, (fit_d.param[5], fit_c.param[5]))

    basis_d = fixed_order_basis(order, fit_d.param)
    basis_c = [(x, y) -> f(x, y) * cis(-(k - 1) * π / 6)
               for (k, f) ∈ enumerate(fixed_order_basis(order, fit_c.param))]

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
h5open("Results/Intense/fixed_order.h5", "cw") do file
    file["mean_fid"] = mean(metrics, dims=2)
    file["std_fid"] = std(metrics, dims=2)
    file["orders"] = collect(orders)
end