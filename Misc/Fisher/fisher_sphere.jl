using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography
includet("../../Utils/basis.jl")
includet("../../Utils/obstructions.jl")
includet("../../Utils/position_operators.jl")

using ColorSchemes

function get_colormap(data)
    colormap = colorschemes[:seismic]
    m, M = extrema(data)

    if abs(M) > abs(m)
        n = round(Int, 51 + 50 * m / M)
        return colormap[n:end]
    else
        n = round(Int, 51 - 50 * M / m)
        return colormap[1:n]
    end
end
##
rs = LinRange(-2.2f0, 2.2f0, 256)
basis_func = positive_l_basis(2, [0.0f0, 0, 1, 1])

povm = assemble_position_operators(rs, rs, basis_func)

prob = StateTomographyProblem(povm)
##
coords = LinRange(-1.0f0, 1, 128)
inv_sqrt_2 = Float32(1 / √2)
θs = [[x * inv_sqrt_2, 0, z * inv_sqrt_2] for x in coords, z ∈ coords]

Is = findall(θ -> sum(abs2, θ) ≤ 1 / 2, θs)

x_coords = [coords[I[1]] for I ∈ Is]
y_coords = [coords[I[2]] for I ∈ Is]

fisher_values = Vector{Float32}(undef, length(Is))

p = Progress(length(Is))
Threads.@threads for n ∈ eachindex(Is)
    I = Is[n]
    θ = θs[I[1], I[2]]
    fisher_values[n] = sum(inv, eigvals(fisher(prob, θ)))
    next!(p)
end

modes = [[abs2(f(x, y)) for x ∈ rs, y ∈ rs] for f in basis_func]
for mode ∈ modes
    normalize!(mode, Inf)
end
modes = vcat(modes[1], modes[2])
##
with_theme(theme_latexfonts()) do
    fig = Figure(size=(600, 700), fontsize=24)

    g = fig[1, 1] = GridLayout()

    ga = g[1, 1] = GridLayout()
    gb = g[2, 1] = GridLayout()

    ax = Axis(ga[1, 1],
        aspect=DataAspect(),
        xlabel=L"x",
        ylabel=L"z",
        xlabelsize=32,
        ylabelsize=32,
        xgridvisible=false,
        ygridvisible=false,
        title="Unobstructed")
    xlims!(ax, (-1, 1))
    ylims!(ax, (-1, 1))


    lim = maximum(abs, fisher_values)
    hm = heatmap!(ax, x_coords, y_coords, fisher_values)
    Colorbar(ga[1, 2], hm, label=L"\text{Tr } I_0^{-1}")

    ax2 = Axis(gb[1, 1], aspect=DataAspect())
    hidedecorations!(ax2)
    hm2 = heatmap!(ax2, modes; colormap=:jet, colorrange=(0, 1))
    Colorbar(gb[1, 2], hm2, label="Intensity (a.u.)")

    rowsize!(g, 2, Auto(0.5))

    #save("Plots/fisher_sphere.png", fig, px_per_unit=4)
    fig
end
##
radius = 1.07f0



I = get_valid_indices(rs, rs, iris_obstruction, 0, 0, radius)

povm = assemble_position_operators(rs, rs, basis_func)[I]

sum(povm)

prob = StateTomographyProblem(povm)

fisher_values_obs = Vector{Float32}(undef, length(Is))

p = Progress(length(Is))
Threads.@threads for n ∈ eachindex(Is)
    I = Is[n]
    θ = θs[I[1], I[2]]
    fisher_values_obs[n] = sum(inv, eigvals(fisher(prob, θ)))
    next!(p)
end

modes = [[abs2(f(x, y)) for x ∈ rs, y ∈ rs] for f in get_obstructed_basis(basis_func, iris_obstruction, 0, 0, radius)]
for mode ∈ modes
    normalize!(mode, Inf)
end
modes = vcat(modes[1], modes[2])

with_theme(theme_latexfonts()) do
    fig = Figure(size=(600, 700), fontsize=24)

    g = fig[1, 1] = GridLayout()

    ga = g[1, 1] = GridLayout()
    gb = g[2, 1] = GridLayout()

    ax = Axis(ga[1, 1],
        aspect=DataAspect(),
        xlabel=L"x",
        ylabel=L"z",
        xlabelsize=32,
        ylabelsize=32,
        xgridvisible=false,
        ygridvisible=false,
        title="Radius = $radius w")
    xlims!(ax, (-1, 1))
    ylims!(ax, (-1, 1))


    log_relative = log2.(fisher_values_obs ./ fisher_values)
    colormap = get_colormap(log_relative)

    lim = maximum(abs, log_relative)
    hm = heatmap!(ga[1, 1], x_coords, y_coords, log_relative; colormap)
    Colorbar(ga[1, 2], hm, label=L"\log_2 \left( \text{Tr } I^{-1} / \text{Tr } I_0^{-1} \right)")

    ax2 = Axis(gb[1, 1], aspect=DataAspect())
    hidedecorations!(ax2)
    hm2 = heatmap!(ax2, modes; colormap=:jet)
    Colorbar(gb[1, 2], hm2, label="Intensity")

    rowsize!(g, 2, Auto(0.5))

    #save("Plots/fisher_sphere_r=$r.png", fig, px_per_unit=4)

    fig
end
##
xb = 0.5f0
I = get_valid_indices(rs, rs, blade_obstruction, xb)

povm = assemble_position_operators(rs, rs, basis_func)[I]
L = transform_incomplete_povm!(povm)
prob = StateTomographyProblem(povm)

fisher_values_obs = Vector{Float32}(undef, length(Is))

p = Progress(length(Is))
Threads.@threads for n ∈ eachindex(Is)
    I = Is[n]
    θ = θs[I[1], I[2]]
    fisher_values_obs[n] = sum(inv, eigvals(incomplete_fisher(prob, θ, L)))
    next!(p)
end

modes = [[abs2(f(x, y)) for x ∈ rs, y ∈ rs] for f in get_obstructed_basis(basis_func, blade_obstruction, xb)]
for mode ∈ modes
    normalize!(mode, Inf)
end
modes = vcat(modes[1], modes[2])

with_theme(theme_latexfonts()) do
    fig = Figure(size=(600, 700), fontsize=24)

    g = fig[1, 1] = GridLayout()

    ga = g[1, 1] = GridLayout()
    gb = g[2, 1] = GridLayout()

    ax = Axis(ga[1, 1],
        aspect=DataAspect(),
        xlabel=L"x",
        ylabel=L"z",
        xlabelsize=32,
        ylabelsize=32,
        xgridvisible=false,
        ygridvisible=false,
        title=L"x_b = %$xb w")
    xlims!(ax, (-1, 1))
    ylims!(ax, (-1, 1))


    log_relative = log2.(fisher_values_obs ./ fisher_values)
    colormap = get_colormap(log_relative)

    lim = maximum(abs, log_relative)
    hm = heatmap!(ga[1, 1], x_coords, y_coords, log_relative; colormap)
    Colorbar(ga[1, 2], hm, label=L"\log_2 \left( \text{Tr } I^{-1} / \text{Tr } I_0^{-1} \right)")

    ax2 = Axis(gb[1, 1], aspect=DataAspect())
    hidedecorations!(ax2)
    hm2 = heatmap!(ax2, modes; colormap=:jet)
    Colorbar(gb[1, 2], hm2, label="Intensity")

    rowsize!(g, 2, Auto(0.5))

    #save("Plots/fisher_sphere_xb=$xb.png", fig, px_per_unit=4)

    fig
end