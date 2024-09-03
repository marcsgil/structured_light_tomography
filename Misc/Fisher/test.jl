using BayesianTomography

includet("../../Utils/basis.jl")
includet("../../Utils/position_operators.jl")
includet("../../Utils/obstructions.jl")

rs = LinRange(-3.0f0, 3.0f0, 256)
basis = positive_l_basis(2, [0, 0, 1, 1])
measurement = assemble_position_operators(rs, rs, basis)
##
θs = Float32.([0, 0, 0] / √2)
ρ = density_matrix_reconstruction(θs)
##
radius = 0.5
J = get_valid_indices(rs, rs, iris_obstruction, 0, 0, radius)
problem = StateTomographyProblem(measurement[J])
method = BayesianInference(problem)
N = 10^4
repetitions = 200
result = Vector{Float32}(undef, repetitions)

C = sum(inv, eigvals(fisher(problem, θs)))
outcomes = simulate_outcomes(get_probabilities(problem, θs), N)
ρ_pred, θ_pred, cov = prediction(outcomes, method)




Threads.@threads for n ∈ eachindex(result)
    C = sum(inv, eigvals(fisher(problem, θs)))

    outcomes = simulate_outcomes(get_probabilities(problem, θs), N)

    ρ_pred, θ_pred, cov = prediction(outcomes, method)

    result[n] = N * sum(abs2, ρ - ρ_pred) / C
end

mean(result)