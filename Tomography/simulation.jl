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

counts = 10:50:1000
error = Matrix{Float64}(undef, size(ρs, 3), length(counts))
##
for n ∈ eachindex(counts)
    for (m, ρ) ∈ enumerate(eachslice(ρs, dims=3))
        sim = simulate_outcomes(ρ, povm, counts[n])
        σ, _ = prediction(sim, mthd)
        error[m, n] = sum(abs2, ρ - σ)
    end
end

error
mean_error = dropdims(mean(error, dims=1), dims=1)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1];
        xlabel="Number of counts",
        ylabel="Error",
        xscale = log10,
        yscale=log10)
    lines!(ax, counts, mean_error)
    lines!(ax, counts, 4.418./counts)
    fig
end