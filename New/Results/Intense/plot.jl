using CairoMakie, HDF5

file = h5open("New/Results/Intense/linear_inversion.h5")
fids_linear = read(file["fids"])
fids_std_linear = read(file["fids_std"])
close(file)

file = h5open("New/Results/Intense/machine_learning.h5")
fids_ml = read(file["fids"])
fids_std_ml = read(file["fids_std"])
close(file)

orders = 1:5
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

    axislegend(ax, position=:rt)
    fig
    save("New/Plots/fidelities_mixed.pdf", fig)
end