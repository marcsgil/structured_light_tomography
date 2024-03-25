using CairoMakie, HDF5

file = h5open("Results/Intense/linear_inversion.h5")
fids_linear = read(file["fids"])
close(file)
##
file = h5open("Results/Intense/machine_learning.h5")
fids_ml = read(file["fids"])
close(file)
##
orders = 1:5
my_theme = Theme(
    fontsize=28,
    markersize=28,
    linewidth=3)
theme = merge(my_theme, theme_latexfonts())

with_theme(theme) do
    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1], xlabel="Order", ylabel="Mean Fidelity", yticks=0.91:0.01:1)
    ylims!(ax, 0.96, 1)

    scatter!(ax, orders, fids_linear, color=:blue, label="Linear inversion")
    scatter!(ax, orders, fids_ml, color=:red, label="Machine learning", marker=:diamond)

    axislegend(ax, position=:lb)
    fig
    #save("Plots/fidelities_mixed.pdf", fig)
end