using BayesianTomography, LinearAlgebra

includet("../utils.jl")

order = 5


basis = transverse_basis(order) |> reverse

rs = LinRange(-5, 5, 128)
direct_operators = assemble_position_operators(rs, rs, basis)
mode_converter = diagm([cis(k * π / 2) for k ∈ 0:order])
astig_operators = assemble_position_operators(rs, rs, basis)
unitary_transform!(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)


angles = random_angles(10)
ψ = hurwitz_parametrization(angles)
rep = vcat(real.(ψ), imag.(ψ))
outcomes = simulate_outcomes(ψ, operators, 2048)

dest = similar(ψ)

@b log_likellyhood($outcomes, $operators, $angles)
@b BayesianTomography.log_likellyhood2!($dest, $outcomes, $operators, $rep)