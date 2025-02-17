using CairoMakie, HDF5

orders, photocounts, fid = h5open("Results/fixed_order_photocount.h5") do file
    read(file["orders"]), read(file["photocounts"]), read(file["fid_no_calib"])
end

fid

with_theme(theme_latexfonts()) do 
    fig = Figure(; fontsize = 24)
    ax = Axis(fig[1, 1]; xlabel = "Photocounts", ylabel = "Mean Fidelity", xscale = log2, xticks=(2.0).^(6:12), yticks = 0.89:0.01:1)
    
    colors = (:blue, :green, :red, :black)
    markers = (:rect, :circle, :diamond, :xcross)

    for n ∈ axes(fid, 3)
        color = colors[n]
        marker = markers[n]
        scatter!(ax, photocounts, fid[1, :, n]; label = "Dimension $(orders[n] + 1)", markersize = 20, color, marker)
        rangebars!(ax, photocounts, fid[2, :, n], fid[3, :, n]; whiskerwidth = 14, linewidth = 4, color)
    end
    ylims!(ax, 0.89, 1)

    axislegend(ax, position = :rb)
    save("../Structured light tomography from position measurements/fidelities_photocount.pdf", fig)
    fig
end