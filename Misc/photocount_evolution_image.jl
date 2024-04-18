using HDF5, CairoMakie, BayesianTomography, Tullio, PositionMeasurements

order = 1
file = h5open("Data/Processed/pure_photocount.h5")
images = read(file["images_order$order"])
labels = read(file["labels_order$order"])
close(file)
##
rs = LinRange(-3.8, 3.8, 64)
idx = 4

fig = Figure(resolution=(900, 300))
axs = [Axis(fig[1, n]) for n in 1:3]
for ax ∈ axs
    aspect = DataAspect()
    hidedecorations!(ax)
end

images1 = images["128_photocounts"]
images2 = images["2048_photocounts"]

is_converted = 2

heatmap!(axs[1], images1[:, :, is_converted, idx], colormap=:hot)
heatmap!(axs[2], images2[:, :, is_converted, idx], colormap=:hot)
heatmap!(axs[3], label2image(labels[:, idx], rs, π / 2)[:, :, is_converted], colormap=:hot)


fig
##
save("Plots/photocount_evolution_image.pdf", fig)
##
order = 4
file = h5open("Data/Processed/mixed_intense.h5")
images = read(file["images_order$order"])
labels = read(file["labels_order$order"])
close(file)
##
rs = LinRange(-3.8, 3.8, 400)
idx = 5

fig = Figure(resolution=(600, 300))
axs = [Axis(fig[1, n]) for n in 1:2]
for ax ∈ axs
    aspect = DataAspect()
    hidedecorations!(ax)
end
is_converted = 2

heatmap!(axs[1], images[:, :, is_converted, idx], colormap=:hot)
heatmap!(axs[2], label2image(labels[:, :, idx], rs, -π / 6)[:, :, is_converted], colormap=:hot)

fig