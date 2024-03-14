using PositionMeasurements, BayesianTomography, Random, Test, LinearAlgebra
Random.seed!(1234)  # Set the seed for reproducibility

@testset "Linear Inversion (Position Operators)" begin
    for order ∈ 1:4
        @info "Testing Order $order"

        basis = transverse_basis(order)
        R = 2.5 + 0.5 * order
        rs = LinRange(-R, R, 64)
        direct_operators = assemble_position_operators(rs, rs, basis)
        mode_converter = diagm([cis(k * π / (order + 1)) for k ∈ 0:order])
        astig_operators = assemble_position_operators(rs, rs, basis)
        unitary_transform!(astig_operators, mode_converter)
        operators = compose_povm(direct_operators, astig_operators)
        li = LinearInversion(operators)
        bi = BayesianInference(operators, 10^5, 10^3)

        N = 5
        for n ∈ 1:N
            ρ = sample_ginibri_state(order + 1)
            θ = π / (order + 1)
            images = label2image(ρ, rs, θ)
            @test fidelity(prediction(images, li), ρ) ≥ 0.995

            normalize!(images, 1)
            simulate_outcomes!(images, 2^15)
            σ, _ = prediction(images, bi)
            @test fidelity(σ, ρ) ≥ 0.99
        end
    end
end