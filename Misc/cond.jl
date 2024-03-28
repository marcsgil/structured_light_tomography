using BayesianTomography, PositionMeasurements, LinearAlgebra

for order ∈ 1:5
    basis = transverse_basis(order)
    R = 2.5 * 0.5 * order
    rs = LinRange(-R, R, 64)

    direct_povm = assemble_position_operators(rs, rs, basis)
    mode_converter = diagm([cis(Float32(k * π / 6)) for k ∈ 0:order])
    astig_povm = assemble_position_operators(rs, rs, basis)
    unitary_transform!(astig_povm, mode_converter)
    povm = compose_povm(direct_povm, astig_povm)

    println("Order $order: $(cond(povm))")
end

for order ∈ 1:5
    basis = transverse_basis(order)
    R = 2.5 * 0.5 * order
    rs = LinRange(-R, R, 64)

    direct_povm = assemble_position_operators(rs, rs, basis)
    mode_converter = diagm([cis(Float32(k * π / 2)) for k ∈ 0:order])
    astig_povm = assemble_position_operators(rs, rs, basis)
    unitary_transform!(astig_povm, mode_converter)
    povm = compose_povm(direct_povm, astig_povm)

    println("Order $order: $(cond(povm))")
end