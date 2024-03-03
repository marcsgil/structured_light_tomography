using LinearAlgebra, BayesianTomography, Symbolics

includet("../New/utils.jl")

function W_representation(Ω::Matrix{T}) where {T}
    @assert size(Ω, 1) == size(Ω, 2) "Ω must be a square matrix"
    d = size(Ω, 1)
    result = Vector{real(T)}(undef, d^2)

    for k ∈ 1:d
        result[k] = real(Ω[k, k])
    end

    counter = d
    for k ∈ 1:d, j ∈ k+1:d
        counter += 1
        result[counter] = √2 * real(Ω[j, k])
        result[counter+d*(d-1)÷2] = √2 * imag(Ω[j, k])
    end

    result
end

function f(operator::Vector{T}, angles) where {T}
    """d = Int(√length(operator))

    result = zero(eltype(operator))

    for k ∈ 1:d
        result += abs2(ψ[k]) * operator[k]
    end

    counter = d
    for k ∈ 1:d, j ∈ k+1:d
        counter += 1
        z = ψ[j] * conj(ψ[k])
        result += √2 * (operator[counter] * real(z) + operator[counter+d*(d-1)÷2] * imag(z))
    end

    log(result)"""

    s1, c1 = sincos(angles[1] / 2)
    s2, c2 = sincos(angles[2])

    log(operator[1] * c1^2 + operator[2] * s1^2 + √2 * s1 * c1 * (operator[3] * c2 + operator[4] * s2))
end

function log_likelihoodtwo(outcomes, operators, angles)
    sum(pair -> pair.second * f(operators[pair.first], angles), pairs(outcomes))
end
##
order = 5
angles = random_angles(2 * order)
ψ = hurwitz_parametrization(angles)

@code_warntype hurwitz_parametrization(angles)

r = LinRange(-5, 5, 64)
basis = [(r, par) -> hg(r[1], r[2], m, order - m) for m ∈ 0:order]

direct_operators = assemble_position_operators(r, r, basis)
mode_converter = diagm([cis(k * π / 2) for k ∈ 0:order])
astig_operators = assemble_position_operators(r, r, basis)
unitary_transform!(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)

outcomes = simulate_outcomes(Vector(ψ), operators, 1024)
log_likellyhood(outcomes, operators, angles)

@benchmark log_likellyhood($outcomes, $operators, $angles)
@benchmark log_likelihoodtwo($outcomes, $operators2, $angles)