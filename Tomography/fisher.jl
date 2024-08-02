using BayesianTomography, StructuredLight,
    PositionMeasurements, LinearAlgebra, CairoMakie,
    Tullio

includet("../Utils/basis.jl")
##
rs = Base.oneto(200)
x₀ = length(rs) ÷ 2
y₀ = length(rs) ÷ 2
w = length(rs) ÷ 8
γ = w / √2

_basis = positive_l_basis(2, [x₀, y₀, γ, 1])
##
cutoffs = LinRange(x₀ - 1.2w, x₀ + 2.5w, 32)

N = 10^3
d = length(basis)
ρs = sample(ProductMeasure(d), N)

Is = Array{Float32,3}(undef, d^2 - 1, d^2 - 1, length(cutoffs))

Threads.@threads for n ∈ eachindex(cutoffs)
    basis = [(x, y) -> f(x, y) * (x < cutoffs[n]) for f ∈ _basis]
    povm = assemble_position_operators_simple(rs, rs, basis)
    Is[:, :, n] = mean(fisher_information(ρs, povm), dims=3)
end

inv_Is = mapslices(inv, Is, dims=(1, 2))
diag_inv_Is = dropdims(mapslices(diag, inv_Is, dims=(1, 2)), dims=2)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1],
        xlabel="Blade position (waist)",
        ylabel="Average inverse Fisher information",
        yscale=log2,
        yticks=[2^n for n ∈ 0:13],
        xticks=-1:0.5:2.5,
    )
    #yscale=log10)
    #ylims!(ax, 1, 100)
    series!(ax, (cutoffs .- x₀) / w, diag_inv_Is,
        labels=[L"(I^{-1})_{XX}", L"(I^{-1})_{YY}", L"(I^{-1})_{ZZ}"],
        color=[:red, :green, :blue],
        linewidth=3,
        linestyle=[:solid, :dash, :dot],)
    axislegend(position=:rt)
    fig
    #save("Plots/fisher_blade.pdf", fig)
end
##
_basis = [(x, y) -> f(x, y) * (x < cutoffs[27]) for f ∈ basis]

povm = assemble_position_operators_simple(rs, rs, _basis)
#@benchmark assemble_position_operators_simple($rs, $rs, $basis)

img = [real(tr(ρs[:, :, 1] * Π)) for Π ∈ povm]

visualize(img)
##
cutoffs = LinRange(0.3w, 2w, 32)

N = 10^3
d = length(basis)
ρs = sample(ProductMeasure(d), N)

Is = Array{Float32,3}(undef, d^2 - 1, d^2 - 1, length(cutoffs))

for (n, cutoff) ∈ enumerate(cutoffs)
    _povm = filter_povm(povm, R -> circ_filter(R, cutoff, x₀, y₀), rs, rs)
    Is[:, :, n] = mean(fisher_information(ρs, _povm), dims=3)
end

inv_Is = mapslices(inv, Is, dims=(1, 2))
diag_inv_Is = dropdims(mapslices(diag, inv_Is, dims=(1, 2)), dims=2)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1],
        xlabel="Iris radius (waist)",
        ylabel="Average inverse Fisher information",
        yscale=log2,
        yticks=[2^n for n ∈ 0:13],
    )
    #yscale=log10)
    #ylims!(ax, 1, 100)
    series!(ax, cutoffs / w, diag_inv_Is,
        labels=[L"(I^{-1})_{XX}", L"(I^{-1})_{YY}", L"(I^{-1})_{ZZ}"],
        color=[:red, :green, :blue],
        linestyle=[:solid, :dash, :dot],
        linewidth=3,)
    axislegend(position=:rt)
    fig
    #save("Plots/fisher_iris.pdf", fig)
end
##
rs = Base.oneto(200)
x₀ = length(rs) ÷ 2
y₀ = length(rs) ÷ 2
w = length(rs) ÷ 4

basis = only_l_basis(2, [x₀, y₀, w, 1])

povm = assemble_position_operators(rs, rs, basis)

N = 10
d = length(basis)
ρs = sample(ProductMeasure(d), N)

inv_I = mean(ρ -> fisher_information(ρ, povm), eachslice(ρs, dims=3)) |> inv

diag(inv_I) |> sort


ρs[:, :, 5]
visualize([real(tr(ρs[:, :, 6] * Π)) for Π ∈ povm])