using HDF5, CairoMakie, BayesianTomography, Tullio

function history2array(history, size)
    result = zeros(eltype(history), size)
    for event ∈ history
        result[event] += 1
    end
    result
end

function get_basis(rs, order)
    stack([
        map(r -> hg([r[1], r[2]], (order - n, n)), Iterators.product(rs, rs))
        for n ∈ 0:order])
end

function superposition(cs, rs)
    basis = get_basis(rs, length(cs) - 1)
    @tullio result[i, j] := basis[i, j, k] * cs[k]
end

order = 4
file = h5open("Data/ExperimentalData/Photocount/datasets.h5")
histories = read(file["histories_order$order"])
coefficients = read(file["coefficients_order$order"])
close(file)
##
rs = LinRange(-3.8, 3.8, 64)
idx = 5

fig = Figure(resolution=(900, 300))
axes = [Axis(fig[1, n]) for n in 1:3]
for ax ∈ axes
    aspect = DataAspect()
    hidedecorations!(ax)
end

sum(outcomes1) .|> Int32
outcomes1 = history2array(view(histories, 1:128, idx), (64, 64, 2))
outcomes2 = history2array(view(histories, 1:2048, idx), (64, 64, 2))

heatmap!(axes[1], outcomes1[:, :, 1], colormap=:hot)
heatmap!(axes[2], outcomes2[:, :, 1], colormap=:hot)
heatmap!(axes[3], superposition(coefficients[:, idx], rs) .|> abs2, colormap=:hot)

fig
##
save("Plots/photocount_evolution_image.pdf", fig)