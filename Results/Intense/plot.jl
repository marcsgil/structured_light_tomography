using CairoMakie, HDF5, BayesianTomography

includet("../../Data/representations.jl")

file = h5open("Results/Intense/linear_inversion.h5")
fids_linear = read(file["fids"])
fids_std_linear = read(file["fids_std"])
close(file)
##
file = h5open("Results/Intense/machine_learning.h5")
orders = 1:5
fids_ml = zeros(length(orders))
fids_std_ml = zeros(length(orders))
for order ∈ orders
    ρs = read(file["labels_order$order"])
    σs = complex_representation(read(file["pred_labels_order$order"]), MixedState())
    _fids = map((ρ, σ) -> fidelity(ρ, project2density(σ)), eachslice(ρs, dims=3), eachslice(σs, dims=3))
    fids_ml[order] = mean(_fids)
    fids_std_ml[order] = std(_fids)
end
fids_ml
##
my_theme = Theme(
    fontsize=28,
    markersize=28,
    linewidth=3)
theme = merge(my_theme, theme_latexfonts())

with_theme(theme) do
    fig = Figure(resolution=(800, 500))
    ax = Axis(fig[1, 1], xlabel="Order", ylabel="Mean Fidelity", yticks=0.91:0.01:1)
    scatter!(ax, orders, fids_linear, color=:blue, label="Linear inversion")
    scatter!(ax, orders, fids_ml, color=:red, label="Machine learning", marker=:diamond)

    errorbars!(ax, orders, fids_linear, fids_std_linear, color=:blue, whiskerwidth=10)
    errorbars!(ax, orders, fids_ml, fids_std_ml, color=:red, whiskerwidth=10)

    axislegend(ax, position=:lb)
    fig
    save("Plots/fidelities_mixed.pdf", fig)
end