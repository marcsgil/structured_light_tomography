using HDF5, BayesianTomography

relu(x::T1, y::T2) where {T1,T2} = x > y ? x - y : zero(promote_type(T1, T2))

include("data_treatment_utils.jl")
includet("../Utils/model_fitting.jl")

calibration = h5open("Data/Raw/fixed_order_intense.h5") do file
    #print(keys(file))
    read(file["calibration"])
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, 1, maximum(calibration), minimum(calibration)])

fit_d = surface_fit(gaussian_model, x, y, calibration[:, :, 1], p0)
fit_c = surface_fit(gaussian_model, x, y, calibration[:, :, 2], p0)

order = 1
images = read(input["images_order$order"])
ρs = read(input["labels_order$order"])

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
    θs = gell_mann_projection(ρ)
    pred_ρ, pred_θs, cov = prediction(probs, mthd)

    metrics[m, n], errors[m, n] = fidelity_metric(θs, pred_θs, cov, 0.95)
end