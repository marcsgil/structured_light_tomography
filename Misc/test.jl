using BayesianTomography, LinearAlgebra

includet("../utils.jl")

order = 4

basis = transverse_basis(order)
R = 2.5 + 0.5 * order
rs = LinRange(-R, R, 64)
direct_operators = assemble_position_operators(rs, rs, basis)
mode_converter = diagm([cis(k * π / (order + 1)) for k ∈ 0:order])
astig_operators = assemble_position_operators(rs, rs, basis)
unitary_transform!(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)
mthd1 = LinearInversion(operators)
mthd2 = BayesianInference(operators, 10^5, 10^3)
hermitian_basis = get_hermitian_basis(order + 1)

N = 10
for n ∈ 1:N
    @info "Testing order $order, iteration $n / $N"
    X = randn(ComplexF64, order + 1, order + 1)
    ρ = X * X' / tr(X * X')
    images = label2image(ρ, rs, π / (order + 1))
    @assert fidelity(prediction(images, mthd1), ρ) ≥ 0.995

    normalize!(images, 1)
    simulate_outcomes!(images, 2^15)
    outcomes = array2dict(images)
    σ = linear_combination(mean(prediction(outcomes, mthd2)), hermitian_basis)
    @show fidelity(σ, ρ)
end