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

    mean_fid, std_fid, mean_fid_no_calib, std_fid_no_calib, h_values = load_data(path, h_key)

    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=24)
        ax = CairoMakie.Axis(fig[1, 1],
            xlabel=h_label,
            ylabel="Mean Fidelity",
            yticks=0.92:0.01:1)

        scatter!(ax, h_values, mean_fid,
            color=:blue,
            label="With Calibration",
            markersize=24,
            marker=:diamond,
        )
        scatter!(ax, h_values, mean_fid_no_calib,
            color=:red,
            label="Without Calibration",
            markersize=24,
            marker=:rect
        )

        errorbars!(ax, h_values, mean_fid, std_fid, color=:blue, whiskerwidth=16, linewidth=4)
        errorbars!(ax, h_values, mean_fid_no_calib, std_fid_no_calib, color=:red, whiskerwidth=16, linewidth=4)

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

    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=24, size=(800, 1400))

        for n ∈ axes(fid, 3)
            ax = CairoMakie.Axis(fig[n, 1],
                xlabel=h_label,
                ylabel="Mean Fidelity",
                xscale=log2,
                yticks=0.9:0.02:1,
                xticks=h_values)

            if n != size(fid, 3)
                hidexdecorations!(ax, grid=false)
            end

            s1 = scatter!(ax, h_values, fid[1, :, n],
                color=:blue,
                label="With Calibration",
                markersize=24,
                marker=:diamond,
            )
            s2 = scatter!(ax, h_values, fid_no_calib[1, :, n],
                color=:red,
                label="Without Calibration",
                markersize=24,
                marker=:rect
            )

            rangebars!(ax, h_values, fid[2, :, n], fid[3, :, n], color=:blue, whiskerwidth=16, linewidth=4)
            rangebars!(ax, h_values, fid_no_calib[2, :, n], fid_no_calib[3, :, n], color=:red, whiskerwidth=16, linewidth=4)

            if minimum(fid_no_calib[1, :, n]) < 0.915
                low = 0.915
            else
                low = nothing
            end

            ylims!(ax; low, high=1.001)
        end

        rowgap!(fig.layout, 1)
        fig
    end
end

make_plot_photocount("Results/fixed_order_photocount.h5", "photocounts", "Photocounts")