using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography, HDF5
includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("../Utils/obstructed_measurements.jl")

extract_θ(ρ, ωs) = stack(real(tr(ρ * ω)) for ω ∈ eachslice(view(ωs, :, :, 2:4), dims=3))
##
rs = LinRange(-2.2f0, 2.2f0, 256)
basis_func = positive_l_basis(2, [0.0f0, 0, 1, 1])
ωs = gell_mann_matrices(2)

ρs = h5open("Data/template.h5") do file
    read(file, "labels_dim2")
end

#ρs = sample(ProductMeasure(2), 100)

θs = stack(extract_θ(ρ, ωs) for ρ ∈ eachslice(ρs, dims=3))

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
        basis_func_obs = get_obstructed_basis(basis_func, iris_obstruction, 0, 0, radius)
        T, Ω, L = assemble_povm_matrix(rs, rs, basis_func_obs)
        ηs = stack(η_func(θ, ωs, L, ωs) for θ ∈ eachslice(θs, dims=2))

        xs = view(ηs, 1, :) * √2
        ys = view(ηs, 2, :) * √2
        zs = view(ηs, 3, :) * √2

        ax = Axis(fig[1, 2n-1], aspect=DataAspect(), xlabel="x", ylabel="z", title="Radius = $radius")
        lines!(ax, x_coords, y_coords, color=:black, linestyle=:dash, linewidth=4)

        sc = scatter!(ax, xs, zs, color=ys, colormap=:viridis, colorrange=(-1, 1), markersize=13)
        Colorbar(fig[1, 2n], sc, label="y")
    end

    save("Plots/eta_map_iris.pdf", fig)

    fig
end
##
xb = [Inf, -0.5f0, -1f0]

with_theme(theme_latexfonts()) do
    fig = Figure(size=(750 * 3, 600), fontsize=36)

    for (n, xb) ∈ enumerate(xb)
        basis_func_obs = get_obstructed_basis(basis_func, blade_obstruction, xb)
        T, Ω, L = assemble_povm_matrix(rs, rs, basis_func_obs)
        ηs = stack(η_func(θ, ωs, L, ωs) for θ ∈ eachslice(θs, dims=2))

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