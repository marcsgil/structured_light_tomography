using BayesianTomography, HDF5
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")

path = "Data/Training/positive_l_no_noise.h5"

images, labels = h5open(path) do file
    file["images"][:, :, :], file["labels"][:, :]
end

labels

R = 3.0f0
rs = LinRange(-R, R, 64)
basis = positive_l_basis(2, [0, 0, 1, 1])
povm = assemble_position_operators(rs, rs, basis)
problem = StateTomographyProblem(povm)
mthd = LinearInversion(problem)

metrics = Vector{Float64}(undef, size(images, 3))

Threads.@threads for m ∈ axes(images, 3)
    probs = images[:, :, m]
    ρ = density_matrix_reconstruction(labels[:, m])
    pred_ρ, pred_θ, cov = prediction(probs, mthd)

    metrics[m] = fidelity(ρ, pred_ρ)
end

mean(metrics)