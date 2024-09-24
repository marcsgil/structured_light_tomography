using HDF5, CairoMakie

function load_data(path, horizontal_key)
    h5open(path) do file
        (
            read(file["fid"]),
            read(file["fid_no_calib"]),
            read(file[horizontal_key])
        )
    end
end

function make_plot_intense(path, h_key, h_label; saving_path="")

    fid, fid_no_calib, h_values = load_data(path, h_key)

    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=24)
        ax = CairoMakie.Axis(fig[1, 1],
            xlabel=h_label,
            ylabel="Mean Fidelity",
            yticks=0.92:0.01:1)

        scatter!(ax, h_values, fid[1, :],
            color=:blue,
            label="With Calibration",
            markersize=24,
            marker=:diamond,
        )
        scatter!(ax, h_values, fid_no_calib[1, :],
            color=:red,
            label="Without Calibration",
            markersize=24,
            marker=:rect
        )

        rangebars!(ax, h_values, fid[2, :], fid[3, :], color=:blue, whiskerwidth=16, linewidth=4)
        rangebars!(ax, h_values, fid_no_calib[2, :], fid_no_calib[3, :], color=:red, whiskerwidth=16, linewidth=4)

        axislegend(ax, position=:lb)

        if !isempty(saving_path)
            save(saving_path, fig)
        end

        fig
    end
end
##
make_plot_intense("Results/fixed_order_intense.h5", "orders", "Order";
    saving_path="Plots/fixed_order_intense_fid.pdf")
##
make_plot_intense("Results/positive_l.h5", "dims", "Dimension";
    saving_path="Plots/fixed_order_intense_fid.pdf")
##
function make_plot_photocount(path, h_key, h_label; saving_path="")
    fid, fid_no_calib, h_values = load_data(path, h_key)

    colors = [:red, :green, :blue, :black]
    markers = [:rect, :diamond, :circle, :utriangle]
    markers2 = [:pentagon, :star5, :xcross, :star8]

    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=24)
        ax = CairoMakie.Axis(fig[1, 1],
            xlabel=h_label,
            ylabel="Mean Fidelity",
            xscale=log2,
            yticks=0.9:0.01:1,
            xticks=h_values)
        ylims!(ax, 0.93, 1)

        for n ∈ axes(fid, 3)
            scatter!(ax, h_values, fid[1, :, n];
                color=colors[n],
                marker=markers[n],
                label="With Calibration",
                markersize=24,
            )
            scatter!(ax, h_values, fid_no_calib[1, :, n];
                color=colors[n],
                marker=markers2[n],
                label="With Calibration",
                markersize=24,
            )

            rangebars!(ax, h_values, fid[2, :, n], fid[3, :, n];
                color=colors[n],
                whiskerwidth=16,
                linewidth=3
            )
        end

        if !isempty(saving_path)
            save(saving_path, fig)
        end

        fig
    end
end

make_plot_photocount("Results/fixed_order_photocount.h5",
    "photocounts", "Photocounts",
    saving_path="Plots/fixed_order_photocount_fid.pdf")