using BayesianTomography

include("../Utils/obstructions.jl")
include("../Utils/basis.jl")
include("../Utils/position_operators.jl")

rs = LinRange(-2.2f0, 2.2f0, 256)

radius = 1f0

I = get_valid_indices(rs, rs, iris_obstruction, 0, 0, radius)

basis_func = positive_l_basis(2, [0.0f0, 0, 1, 1])

povm = assemble_position_operators(rs, rs, basis_func)[I]

problem = StateTomographyProblem(povm)
θs = zeros(3)

fisher(problem, θs)
#@code_warntype fisher(problem, θs)
##

@benchmark fisher($problem, $θs)
##
σ = density_matrix_reconstruction(θs)
A = problem.kraus_operator
BayesianTomography.kraus_transformation!(σ, A)
N = real(tr(σ))

ωs = GellMannMatrices(problem.dim, eltype(σ))
Ωs = [BayesianTomography.kraus_transformation!(ω, A) for ω ∈ ωs]

J = [real(tr(Ω * ω) / N - tr(σ * ω) * tr(Ω) / N^2) for ω ∈ ωs, Ω ∈ Ωs]

probabilities = Vector{eltype(θs)}(undef, size(problem.traceless_part, 1))
get_probabilities!(probabilities, problem, θs)

ωs = collect(GellMannMatrices(2))

g = sum(povm)
ρ = density_matrix_reconstruction(θs)


[sum(m-> real( tr(povm[m] * ω1) * tr(povm[m] * ω2) ) / probabilities[m], eachindex(probabilities)) for ω1 ∈ ωs, ω2 ∈ ωs] / tr(ρ * g)^2