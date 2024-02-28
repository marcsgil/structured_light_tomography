using BayesianTomography, LinearAlgebra

angles = rand(8) * π

hurwitz_parametrization(angles)
@benchmark hurwitz_parametrization($angles)
@benchmark BayesianTomography.hurwitz_parametrization2($angles)
##
mean = 0.0
N = 100
nsamples = 10000

for _ ∈ 1:nsamples
    #ψ₁ = random_angles(2(N - 1)) |> hurwitz_parametrization
    #ψ₂ = random_angles(2(N - 1)) |> hurwitz_parametrization
    ψ₁ = sample_haar_vector(N)
    ψ₂ = sample_haar_vector(N)
    mean += abs(dot(ψ₁, ψ₂))^2 / nsamples
end

mean
##


sc = sincos.(angles[1:4]) |> stack
s = sc[1, :]

cumprod(s)


cumprod!
##
using Random

function sample_spherical(n_samples::Int)
    phi = 2 * π * rand(n_samples)  # azimuthal angle
    cos_theta = 2 * rand(n_samples) .- 1  # cos(polar angle)
    theta = acos.(cos_theta)  # polar angle

    return theta, phi
end

theta, phi = sample_spherical(1000)

mean(θ -> cos(θ)^2, theta)