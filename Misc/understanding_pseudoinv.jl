using BayesianTomography, LinearAlgebra, CairoMakie

order = 1
r = LinRange(-3, 3, 32)

direct_operators = assemble_position_operators(r, r, order)
mode_converter = diagm([cis(k * π / 6) for k ∈ 0:order])
astig_operators = assemble_position_operators(r, r, order)
astig_operators = unitary_transform(astig_operators, mode_converter)
povm = compose_povm(direct_operators, astig_operators)

basis = BayesianTomography.get_basis(order + 1)
##
A = [real(tr(E * Ω)) for E ∈ vec(povm), Ω ∈ vec(basis)]
image(A * inv(A' * A) * A')