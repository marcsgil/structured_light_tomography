using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography
includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("../Utils/obstructed_measurements.jl")
##
rs = Base.oneto(200)
x₀ = length(rs) ÷ 2
y₀ = length(rs) ÷ 2
w = length(rs) ÷ 8

basis = positive_l_basis(2, [x₀, y₀, w, 1])
d = length(basis)
ρs = sample(ProductMeasure(d), 100)

povm = assemble_position_operators(rs, rs, basis)
mthd = BayesianInference(povm)

counts = [round(Int, 10^n) for n ∈ LinRange(2, 3, 30)]
error_clean = Matrix{Float64}(undef, size(ρs, 3), length(counts))
##
@showprogress for n ∈ eachindex(counts)
    for (m, ρ) ∈ enumerate(eachslice(ρs, dims=3))
        sim = simulate_outcomes(ρ, povm, counts[n])
        σ, _ = prediction(sim, mthd)
        error_clean[m, n] = sum(abs2, ρ - σ)
    end
end

mean_error_clean = dropdims(mean(error_clean, dims=1), dims=1)
##
rs = Base.oneto(200)
x₀ = length(rs) ÷ 2
y₀ = length(rs) ÷ 2
w = length(rs) ÷ 8
cutoff = w / 2

obstructed_basis = get_obstructed_basis(basis, iris_obstruction, x₀, y₀, cutoff)
obstructed_povm, new_ρs = get_proper_povm_and_states(rs, rs, ρs, obstructed_basis)

mthd = BayesianInference(obstructed_povm)

error_obstructed = Matrix{Float64}(undef, size(ρs, 3), length(counts))
##
@showprogress for n ∈ eachindex(counts)
    for (m, ρ) ∈ enumerate(eachslice(new_ρs, dims=3))
        sim = simulate_outcomes(ρ, obstructed_povm, counts[n])
        σ, _ = prediction(sim, mthd)
        error_obstructed[m, n] = sum(abs2, ρ - σ)
    end
end

mean_error_obstructed = dropdims(mean(error_obstructed, dims=1), dims=1)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1];
        xlabel="Number of counts",
        ylabel="Mean squared error",
        xscale=log10,
        yscale=log10)
    scatter!(ax, counts, mean_error_clean, label = "Unobstructed")
    lines!(ax, counts, 4.418 ./ counts)
    scatter!(ax, counts, mean_error_obstructed, label = "Obstructed (r=0.5w)")
    lines!(ax, counts, 3 ./ counts)
    axislegend()
    fig
    save("Plots/simulated_tomography_obstruction.pdf", fig)
end
