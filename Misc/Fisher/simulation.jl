using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography
includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("../Utils/obstructed_measurements.jl")

function covariance_bound_and_probs(θ, mthd, L)
    ω = gell_mann_matrices(mthd.dim)
    η = η_func(θ, ω, L, ω)
    J = η_func_jac(θ, ω, L, ω)
    tr(inv(J' * fisher(mthd, η) * J)), get_probs(mthd, η)
end

function simulate_tomography(θ, mthd, L, counts, repetitions)
    C, probs = covariance_bound_and_probs(θ, mthd, L)
    ω = gell_mann_matrices(mthd.dim)
    θ₀ = convert(eltype(θ), 1 / √mthd.dim)
    ρ = linear_combination(vcat(θ₀, θ), ω)

    error = zeros(Float32, length(counts), repetitions)
    p = Progress(length(error))

    Threads.@threads for n ∈ axes(error, 2)
        for m ∈ axes(error, 1)
            sim = simulate_outcomes(probs, counts[m])
            σ_pred, η_pred, _ = prediction(sim, mthd)
            error[m, n] = sum(abs2, ρ - σ_pred / tr(σ_pred))
            next!(p)
        end
    end

    dropdims(mean(error, dims=2), dims=2), C
end
##
rs = LinRange(-2.0f0, 2.0f0, 256)
θ = Float32(1 / √2) * [0, 0, 0]
counts = [round(Int, 10.0^k) for k ∈ LinRange(2, 4, 20)]
basis_func = positive_l_basis(2, [0.0f0, 0, 1, 1])
##
radius = Float32[Inf, 1.2, 0.8, 0.5]
Cs = similar(radius)
mean_error = similar(radius, Vector{Float32})

for n ∈ eachindex(radius, Cs, mean_error)
    obstructed_basis = get_obstructed_basis(basis_func, iris_obstruction, 0, 0, radius[n])
    T, Ω, L = assemble_povm_matrix(rs, rs, obstructed_basis)
    mthd = BayesianInference(T, Ω)

    mean_error[n], Cs[n] = simulate_tomography(θ, mthd, L, counts, 200)
end
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1];
        xlabel="Number of counts",
        ylabel="Mean squared error",
        xscale=log10,
        yscale=log10)

    for n ∈ eachindex(mean_error, Cs)
        lines!(ax, counts, Cs[n] ./ counts, label="r=$(radius[n])")
        scatter!(ax, counts, mean_error[n])
    end
    axislegend(ax)
    save("Plots/simulated_tomography_obstruction.pdf", fig)
    fig
end
##
using LsqFit

model(N, p) = log(p[1]) .- log.(N)

fit = LsqFit.curve_fit(model, counts, log.(mean_error_ob), [5.0])

confidence_interval(fit, 0.05)