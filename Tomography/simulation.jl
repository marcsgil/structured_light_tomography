using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography
includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("../Utils/obstructed_measurements.jl")
includet("../Utils/fisher.jl")
##
rs = LinRange(-2f0, 2f0, 512)

basis_func = positive_l_basis(2, [0.0f0, 0, 1, 1])
d = length(basis_func)

θ = Float32(1 / √2) * [1, 0, 0, 0]
ρ = linear_combination(θ, gell_mann_matrices(2))
img = get_intensity(ρ, basis_func, rs, rs)
normalize!(img, 1)

visualize(img)

_T, Ω, L, p_corr = assemble_povm_matrix(basis_func, rs, rs)
T = hcat(p_corr * Float32(√2), _T)
mthd = BayesianInference(T, Ω)
ω = gell_mann_matrices(2)

counts = [round(Int, 10^n) for n ∈ LinRange(3, 4, 20)]
error_clean = Matrix{Float32}(undef, length(counts), 10^2)

I0 = fisher_at(θ, basis_func, Ω, L, ω, rs, _T)
##
p = Progress(length(error_clean))
Threads.@threads for n ∈ axes(error_clean, 2)
    for m ∈ axes(error_clean, 1)
        sim = simulate_outcomes(img, counts[m])
        σ, θ_pred, _ = prediction(sim, mthd)
        #error_clean[m, n] = sum(abs2, ρ - ϕ * ϕ' )
        #ϕ = project2pure(σ / tr(σ))
        error_clean[m, n] = sum(abs2, ρ - σ / tr(σ))
        next!(p)
    end
end

mean_error_clean = dropdims(mean(error_clean, dims=2), dims=2)
##
obstructed_basis = [(x, y) -> f(x, y) * iris_obstruction(x, y, 0, 0, 0.1f0) for f in basis_func]

img = get_intensity(ρ, obstructed_basis, rs, rs)
normalize!(img, 1)

visualize(img) |> display

_T, Ω, L, p_corr = assemble_povm_matrix(obstructed_basis, rs, rs)
T = hcat(p_corr * Float32(√2), _T)
mthd = BayesianInference(T, Ω)
ω = gell_mann_matrices(2)

Ω

error_ob = similar(error_clean)


I_ob = fisher_at(θ, obstructed_basis, Ω, L, ω, rs, _T)

η_func(θ, Ω, L, ω)
##
sim = simulate_outcomes(img, 10^4)
σ, θ_pred, _ = prediction(sim, mthd)

σ / tr(σ)
##
p = Progress(length(error_ob))
Threads.@threads for n ∈ axes(error_ob, 2)
    for m ∈ axes(error_ob, 1)
        sim = simulate_outcomes(img, counts[m])
        σ, θ_pred, _ = prediction(sim, mthd)
        error_ob[m, n] = sum(abs2, ρ - σ / tr(σ))
        next!(p)
    end
end

mean_error_ob = dropdims(mean(error_ob, dims=2), dims=2)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1];
        xlabel="Number of counts",
        ylabel="Mean squared error",
        xscale=log10,
        yscale=log10)
    scatter!(ax, counts, mean_error_clean, label="Unobstructed")
    lines!(ax, counts, I0 ./ counts)
    scatter!(ax, counts, mean_error_ob, label = "Obstructed (r=0.5w)")
    lines!(ax, counts, I_ob ./ counts)
    axislegend()
    fig
    #save("Plots/simulated_tomography_obstruction.pdf", fig)
end
##
using LsqFit

model(N, p) = log(p[1]) .- log.(N) 

fit = LsqFit.curve_fit(model, counts, log.(mean_error_ob), [5.])

confidence_interval(fit, 0.05)