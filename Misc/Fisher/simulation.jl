using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography
includet("../../Utils/basis.jl")
includet("../../Utils/obstructions.jl")
includet("../../Utils/position_operators.jl")

function simulate_tomography(θ, mthd, counts, repetitions)
    probs = get_probabilities(mthd.problem, θ)
    ρ = density_matrix_reconstruction(θ)
    C = sum(inv, eigvals(fisher(mthd.problem, θ)))

    error = zeros(Float32, length(counts), repetitions)
    p = Progress(length(error))

    Threads.@threads for n ∈ axes(error, 2)
        for m ∈ axes(error, 1)
            sim = simulate_outcomes(probs, counts[m])
            ρ_pred, _ = prediction(sim, mthd)
            error[m, n] = real(tr((ρ - ρ_pred)^2))
            next!(p)
        end
    end

    dropdims(mean(error, dims=2), dims=2), C
end
##
rs = LinRange(-2.0f0, 2.0f0, 128)
θ = Float32(1 / √2) * [0, 0, 0]
counts = [round(Int, 10.0^k) for k ∈ LinRange(2, 4, 20)]
basis_func = positive_l_basis(2, [0.0f0, 0, 1, 1])
measurement = assemble_position_operators(rs, rs, basis_func)
##
radius = Float32[Inf, 1.2, 0.8, 0.5]
#radius = Float32[Inf, 0.5]
Cs = similar(radius)
mean_error = similar(radius, Vector{Float32})

for n ∈ eachindex(radius, Cs, mean_error)
    I = get_valid_indices(rs, rs, iris_obstruction, 0, 0, radius[n])
    problem = StateTomographyProblem(measurement[I])
    mthd = BayesianInference(problem)

    mean_error[n], Cs[n] = simulate_tomography(θ, mthd, counts, 200)
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
    #save("Plots/simulated_tomography_obstruction.pdf", fig)
    fig
end
##
using LsqFit

model(N, p) = log(p[1]) .- log.(N)

fit = LsqFit.curve_fit(model, counts, log.(mean_error_ob), [5.0])

confidence_interval(fit, 0.05)