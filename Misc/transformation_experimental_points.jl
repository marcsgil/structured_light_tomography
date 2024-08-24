using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography, HDF5
includet("../Utils/basis.jl")
includet("../Utils/incomplete_measurements.jl")
includet("../Utils/position_operators.jl")
##
rs = LinRange(-2.2f0, 2.2f0, 256)
basis_func = positive_l_basis(2, [0.0f0, 0, 1, 1])

ρs = h5open("Data/template.h5") do file
    read(file, "labels_dim2")
end

θs = stack(gell_mann_projection(ρ) for ρ ∈ eachslice(ρs, dims=3))

angles = LinRange(0, 2π, 200)
x_coords = cos.(angles)
y_coords = sin.(angles)
##
with_theme(theme_latexfonts()) do
    xs = view(θs, 1, :) * √2
    ys = view(θs, 2, :) * √2
    zs = view(θs, 3, :) * √2

    fig = Figure(size=(750, 600), fontsize=32)
    ax = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="z")
    lines!(ax, x_coords, y_coords, color=:black, linestyle=:dash, linewidth=4)

    sc = scatter!(ax, xs, zs, color=ys, colormap=:viridis, colorrange=(-1, 1), markersize=13)
    Colorbar(fig[1, 2], sc, label="y")
    fig
end
##
radius = [Inf, 0.75f0, 0.5f0]

with_theme(theme_latexfonts()) do
    fig = Figure(size=(750 * 3, 600), fontsize=36)

    for (n, radius) ∈ enumerate(radius)
        I = get_valid_indices(rs, rs, iris_obstruction, 0, 0, radius)
        povm = assemble_position_operators(rs, rs, basis_func)[I]
        L = transform_incomplete_povm!(povm)
        ηs = stack(η_func(θ, L) for θ ∈ eachslice(θs, dims=2))

        xs = view(ηs, 1, :) * √2
        ys = view(ηs, 2, :) * √2
        zs = view(ηs, 3, :) * √2

        ax = Axis(fig[1, 2n-1], aspect=DataAspect(), xlabel="x", ylabel="z", title="Radius = $radius")
        lines!(ax, x_coords, y_coords, color=:black, linestyle=:dash, linewidth=4)

        sc = scatter!(ax, xs, zs, color=ys, colormap=:viridis, colorrange=(-1, 1), markersize=13)
        Colorbar(fig[1, 2n], sc, label="y")
    end

    #save("Plots/eta_map_iris.pdf", fig)

    fig
end
##
xb = [Inf, -0.5f0, -1.0f0]

I = get_valid_indices(rs, rs, blade_obstruction, xb[3])

with_theme(theme_latexfonts()) do
    fig = Figure(size=(750 * 3, 600), fontsize=36)

    for (n, xb) ∈ enumerate(xb)
        I = get_valid_indices(rs, rs, blade_obstruction, xb)
        povm = assemble_position_operators(rs, rs, basis_func)[I]
        L = transform_incomplete_povm!(povm)
        ηs = stack(η_func(θ, L) for θ ∈ eachslice(θs, dims=2))

        xs = view(ηs, 1, :) * √2
        ys = view(ηs, 2, :) * √2
        zs = view(ηs, 3, :) * √2

        ax = Axis(fig[1, 2n-1], aspect=DataAspect(), xlabel="x", ylabel="z", title="Blade pos. = $xb")
        lines!(ax, x_coords, y_coords, color=:black, linestyle=:dash, linewidth=4)

        sc = scatter!(ax, xs, zs, color=ys, colormap=:viridis, colorrange=(-1, 1), markersize=13)
        Colorbar(fig[1, 2n], sc, label="y")
    end

    #save("Plots/eta_map_blade.pdf", fig)

    fig
end