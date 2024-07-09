using BayesianTomography, StructuredLight,
    PositionMeasurements, LinearAlgebra, CairoMakie,
    Tullio

function filter_povm(povm, filter, x, y)
    Rs = collect(Iterators.product(x, y))
    idxs = findall(filter, Rs)
    povm[idxs]
end

h_filter(R, cutoff) = R[1] > cutoff
circ_filter(R, cutoff) = sqrt(R[1]^2 + R[2]^2) < cutoff
##
rs = LinRange(-4, 4, 256)
#cutoffs = LinRange(-3, 2, 16)
cutoffs = LinRange(0.1, 4, 16)

basis = [(r, par) -> lg(r[1], r[2]; p=1, l=0),
    (r, par) -> lg(r[1], r[2]; l=2)]
povm = assemble_position_operators(rs, rs, basis)

N = 10^3
d = length(basis)
ρs = sample(GinibreEnsamble(d), N)

Is = Array{Float32,3}(undef, d^2 - 1, d^2 - 1, length(Rs))

for (n, cutoff) ∈ enumerate(cutoffs)
    _povm = filter_povm(povm, R -> circ_filter(R, cutoff), rs, rs)
    Is[:, :, n] = mean(fisher_information(ρs, _povm), dims=3)
end

@tullio diag_Is[i, j] := Is[i, i, j]
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1],
        xlabel="Blade position (waist)",
        ylabel="Average Fisher information",
        xticks=first(Rs):last(Rs),
        yticks=0:0.1:2,
        title="MUB",
    )
    #yscale=log10)
    #ylims!(ax, 0, 1.3)
    series!(ax, cutoffs, diag_Is,
        labels=[L"I_{XX}", L"I_{YY}", L"I_{ZZ}"],
        color=[:red, :green, :blue],
        linewidth=3,)
    axislegend()
    fig
    #save("Plots/fisher_mub.png", fig)
end