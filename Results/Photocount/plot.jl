using CairoMakie, HDF5

file = h5open("Results/Photocount/bayes.h5")
fids_bayes = read(file["fids"])
close(file)

file = h5open("Results/Photocount/machine_learning.h5")
fids_ml = read(file["fids"])
fids_std_ml = read(file["fids_std"])
close(file)

photocounts = [2^k for k ∈ 6:11]
##
my_theme = Theme(
    fontsize=28,
    markersize=28,
    linewidth=3)
theme = merge(my_theme, theme_latexfonts())

order = 2

fids_ml

with_theme(theme) do
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1],
        xlabel="Photocounts",
        ylabel="Mean Fidelity",
        yticks=0.88:0.01:1,
        xticks=photocounts,
        xscale=log2)

    ylims!(ax, 0.88, 1)
    color = [:red, :blue, :green, :black]
    markers = [:circle, :diamond, :utriangle, :rect]

    for (fid_ml, fid_bayes, color, order, marker) ∈ zip(eachslice(fids_ml, dims=2), eachslice(fids_bayes, dims=2), color, 1:4, markers)
        lines!(ax, photocounts, fid_ml; color)
        lines!(ax, photocounts, fid_bayes; color, linestyle=:dash)
        scatter!(ax, photocounts, fid_ml;
            marker, color, label="Order $order")
        scatter!(ax, photocounts, fid_bayes;
            marker, color)
    end
    axislegend(ax, position=:rb)
    fig
    #save("Plots/fidelities_photocount.pdf", fig)
end