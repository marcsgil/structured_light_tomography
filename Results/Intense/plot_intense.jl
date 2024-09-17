using HDF5, CairoMakie

function load_data(path, horizontal_key)
    h5open(path) do file
        (
            vec(read(file["mean_fid"])),
            vec(read(file["std_fid"])),
            read(file[horizontal_key])
        )
    end
end

function make_plot(mean_fid, std_fid, mean_fid_ml, std_fid_ml, h_values, h_label;
    saving_path="")
    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=24)
        ax = CairoMakie.Axis(fig[1, 1], 
        xlabel=h_label, 
        ylabel="Mean Fidelity",
        yticks = 0.95:0.01:1)

        scatter!(ax, h_values, mean_fid,
            color=:blue,
            label="Least Squares",
            markersize=24,
            marker=:diamond,
        )
        scatter!(ax, h_values, mean_fid_ml,
            color=:red,
            label="Neural Network",
            markersize=24,
            marker=:rect
        )

        errorbars!(ax, h_values, mean_fid, std_fid, color=:blue, whiskerwidth=16, linewidth=4)
        errorbars!(ax, h_values, mean_fid_ml, std_fid_ml, color=:red, whiskerwidth=16, linewidth=4)

        axislegend(ax, position=:lb)

        if !isempty(saving_path)
            save(saving_path, fig)
        end

        fig
    end
end
##
mean_fid, std_fid, orders = load_data("Results/Intense/fixed_order.h5", "orders")
mean_fid_ml, std_fid_ml, orders_ml = load_data("Results/Intense/fixed_order_with_ml.h5", "orders")

make_plot(mean_fid, std_fid, mean_fid_ml, std_fid_ml, orders, "Order",
    saving_path="Plots/fixed_order_fid.pdf")
##
mean_fid, std_fid, orders = load_data("Results/Intense/positive_l.h5", "dims")
mean_fid_ml, std_fid_ml, orders_ml = load_data("Results/Intense/positive_l_with_ml.h5", "dims")

make_plot(mean_fid, std_fid, mean_fid_ml, std_fid_ml, orders, "Dimension",
    saving_path="Plots/positive_l_fid.pdf")