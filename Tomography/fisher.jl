using BayesianTomography, StructuredLight,
    PositionMeasurements, LinearAlgebra, CairoMakie,
    Tullio

includet("../Utils/basis.jl")

function filter_povm(povm, filter, x, y)
    Rs = collect(Iterators.product(x, y))
    idxs = findall(filter, Rs)
    povm[idxs]
end

h_filter(R, cutoff) = R[1] < cutoff
circ_filter(R, cutoff, x₀, y₀) = sqrt((R[1] - x₀)^2 + (R[2] - y₀)^2) < cutoff
##
rs = Base.oneto(200)
x₀ = length(rs) ÷ 2
y₀ = length(rs) ÷ 2
w = length(rs) ÷ 8

basis = positive_l_basis(2, [x₀, y₀, w, 1])
povm = assemble_position_operators(rs, rs, basis)
##
cutoffs = LinRange(x₀ - 1.2w, x₀ + 2.5w, 32)

N = 10^3
d = length(basis)
ρs = sample(ProductMeasure(d), N)

Is = Array{Float32,3}(undef, d^2 - 1, d^2 - 1, length(cutoffs))

for (n, cutoff) ∈ enumerate(cutoffs)
    _povm = filter_povm(povm, R -> h_filter(R, cutoff), rs, rs)
    Is[:, :, n] = mean(fisher_information(ρs, _povm), dims=3)
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
    save("Plots/fisher_blade.pdf", fig)
end
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