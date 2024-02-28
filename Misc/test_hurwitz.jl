using BayesianTomography, LinearAlgebra

angles = rand(8) * π

hurwitz_parametrization(angles)
@benchmark hurwitz_parametrization($angles)
@benchmark BayesianTomography.hurwitz_parametrization2($angles)
##
mean = 0.0
N = 2
nsamples = 10000

for _ ∈ 1:nsamples
    ψ₁ = random_angles(2(N - 1)) |> hurwitz_parametrization
    ψ₂ = random_angles(2(N - 1)) |> hurwitz_parametrization
    mean += abs(dot(ψ₁, [1 / √2, 1 / √2]))^2 / nsamples
end

mean
##


sc = sincos.(angles[1:4]) |> stack
s = sc[1, :]

cumprod(s)


cumprod!