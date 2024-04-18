using HDF5, CairoMakie, BayesianTomography, Tullio, PositionMeasurements

order = 4
file = h5open("Data/Processed/pure_photocount.h5")
histories = file["histories_order$order"] |> read
labels = read(file["labels_order$order"])
close(file)
##
rs = LinRange(-3.8, 3.8, 64)
idx = 5
observations = Observable(1)
is_converted = 1
possible_obs = Vector(0:8:2048)
possible_obs[1] = 1

with_theme(theme_latexfonts()) do

    fig = Figure(size=(1000, 530), fontsize=20)
    ax = Axis(fig[1, 1], aspect=1)
    ax2 = Axis(fig[1, 2], aspect=1, title="Intense")

    hidedecorations!(ax)
    hidedecorations!(ax2)

    heatmap!(ax2, label2image(labels[:, idx], rs, π / 2)[:, :, is_converted], colormap=:hot)

    image = lift(observations) do obs
        complete_representation(History(view(histories, 1:obs, idx)), (64, 64, 2))[:, :, is_converted]
    end

    heatmap!(ax, image, colormap=:hot)

    framerate = 30
    timestamps = range(0, 2, step=1 / framerate)

    record(fig, "Plots/photocount_animation.mp4", possible_obs;
        framerate=framerate) do t
        observations[] = t
        ax.title = "Photocounts: $t"
    end


end