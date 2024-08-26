using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra, FiniteDiff
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/model_fitting.jl")
includet("../Data/data_treatment_utils.jl")

input = h5open("Data/Processed/mixed_intense.h5")

direct_lims = read(input["direct_lims"])
converted_lims = read(input["converted_lims"])

xd, yd = get_grid(direct_lims, (400, 400))
xc, yc = get_grid(converted_lims, (400, 400))
weights = read(input["weights"])
##
orders = 1:5
fids = Matrix{Float64}(undef, length(orders), 100)
errors = Matrix{Float64}(undef, length(orders), 100)

function BayesianTomography.fidelity(ρ, θ)
    σ = density_matrix_reconstruction(θ)
    fidelity(ρ, σ)
end

@showprogress for (m, order) ∈ enumerate(orders)
    images = read(input["images_order$order"])
    ρs = read(input["labels_order$order"])
    basis = fixed_order_basis(order, [0, 0, √2, 1])

    direct_povm = assemble_position_operators(xd, yd, basis)
    converted_povm = assemble_position_operators(xc, yc, basis)
    mode_converter = diagm([cis(-k * π / 6) for k ∈ 0:order])
    unitary_transform!(converted_povm, mode_converter)
    povm = compose_povm(direct_povm, converted_povm)
    problem = StateTomographyProblem(povm)
    mthd = LinearInversion(problem)

    for n ∈ axes(images, 4)
        probs = images[:, :, :, n]
        ρ = ρs[:, :, n]
        pred_ρ, pred_θs, cov = prediction(probs, mthd)
        θs = gell_mann_projection(ρ)
        δ = pred_θs - θs
        fids[m, n] = fidelity(ρ, pred_θs)
        grad = FiniteDiff.finite_difference_gradient(θ -> fidelity(ρ, θ), pred_θs)
        errors[m, n] = dot(grad, cov, grad) * 1.96
        #fids[m, n] = sum(abs2, δ)
        #errors[m, n] = sqrt(2 * cov ⋅ cov) * 1.96
    end
end

dropdims(mean(fids, dims=2), dims=2)
dropdims(mean(errors, dims=2), dims=2)
##

order = 1

images = read(input["images_order$order"])
ρs = read(input["labels_order$order"])
basis = fixed_order_basis(order, [0, 0, √2, 1])

direct_povm = assemble_position_operators(xd, yd, basis)
converted_povm = assemble_position_operators(xc, yc, basis)
mode_converter = diagm([cis(-k * π / 6) for k ∈ 0:order])
unitary_transform!(converted_povm, mode_converter)
povm = compose_povm(direct_povm, converted_povm)
problem = StateTomographyProblem(povm)
mthd = LinearInversion(problem)

n=1
probs = images[:, :, :, n]
ρ = ρs[:, :, n]
pred_ρ, pred_θs, cov = prediction(probs, mthd)
θs = gell_mann_projection(ρ)

cov



δ = pred_θs - θs
#fids[m, n] = fidelity(ρs[:, :, n], σ)
sum(abs2, δ)


probs = images[:, :, :, n]
pred_probs = reshape(get_probabilities(problem, pred_θs), size(probs))

√sum(abs2, probs-pred_probs)

x = rand(size(probs)...)
y = rand(size(probs)...)

normalize!(x, 1)
normalize!(y, 1)

sum(abs, x-y)

visualize(probs-pred_probs)



errors[m, n] = BayesianTomography.sum_residues(probs, mthd, pred_θs, 1 / sum(probs))

##
out = h5open("New/Results/Intense/linear_inversion.h5", "w")
out["fids"] = dropdims(mean(fids, dims=2), dims=2)
out["fids_std"] = dropdims(std(fids, dims=2), dims=2)
close(out)