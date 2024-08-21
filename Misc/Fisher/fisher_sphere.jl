using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography
includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("../Utils/obstructed_measurements.jl")

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
ωs = gell_mann_matrices(2)
##
T, Ω, L = assemble_povm_matrix(rs, rs, basis_func)
mthd = LinearInversion(T, Ω)

coords = LinRange(-1.0f0, 1, 512)
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
    η = η_func(θ, ωs, L, ωs)
    J = η_func_jac(θ, ωs, L, ωs)
    fisher_values[n] = tr(inv(J' * fisher(mthd, η) * J))
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
        title = "Unobstructed")
    xlims!(ax, (-1, 1))
    ylims!(ax, (-1, 1))


    lim = maximum(abs, fisher_values)
    hm = heatmap!(ax, x_coords, y_coords, fisher_values)
    Colorbar(ga[1, 2], hm, label=L"\text{Tr } I_0^{-1}")

    ax2 = Axis(gb[1, 1], aspect=DataAspect())
    hidedecorations!(ax2)
    hm2 = heatmap!(ax2, modes; colormap=:jet, colorrange=(0,1))
    Colorbar(gb[1, 2], hm2, label="Intensity (a.u.)")

    rowsize!(g, 2, Auto(0.5))

    #save("Plots/fisher_sphere.png", fig, px_per_unit=4)
    fig
end
##
ωs = gell_mann_matrices(2)
r = 0.25f0
basis_func_obs = [(x, y) -> f(x, y) * iris_obstruction(x, y, 0, 0, r) for f in basis_func]
T, Ω, L = assemble_povm_matrix(rs, rs, basis_func_obs)
mthd = LinearInversion(T, Ω)

fisher_values_obs = Vector{Float32}(undef, length(Is))

p = Progress(length(Is))
Threads.@threads for n ∈ eachindex(Is)
    I = Is[n]
    θ = θs[I[1], I[2]]
    η = η_func(θ, ωs, L, ωs)
    J = η_func_jac(θ, ωs, L, ωs)
    fisher_values_obs[n] = tr(inv(J' * fisher(mthd, η) * J))
    next!(p)
end

modes = [[abs2(f(x, y)) for x ∈ rs, y ∈ rs] for f in basis_func_obs]
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
        title = "Radius = $r w")
    xlims!(ax, (-1, 1))
    ylims!(ax, (-1, 1))


    log_relative = log2.(fisher_values_obs ./ fisher_values)
    colormap = get_colormap(log_relative)

    lim = maximum(abs, log_relative)
    hm = heatmap!(ga[1,1], x_coords, y_coords, log_relative; colormap)
    Colorbar(ga[1,2], hm, label=L"\log_2 \left( \text{Tr } I^{-1} / \text{Tr } I_0^{-1} \right)")

    ax2 = Axis(gb[1, 1], aspect=DataAspect())
    hidedecorations!(ax2)
    hm2 = heatmap!(ax2, modes; colormap=:jet)
    Colorbar(gb[1, 2], hm2, label="Intensity")

    rowsize!(g, 2, Auto(0.5))

    #save("Plots/fisher_sphere_r=$r.png", fig, px_per_unit=4)

    fig
end
##
ωs = gell_mann_matrices(2)
xb = -1f0
basis_func_obs = [(x, y) -> f(x, y) * blade_obstruction(x, y, xb) for f in basis_func]
T, Ω, L = assemble_povm_matrix(rs, rs, basis_func_obs)
mthd = LinearInversion(T, Ω)

fisher_values_obs = Vector{Float32}(undef, length(Is))

p = Progress(length(Is))
Threads.@threads for n ∈ eachindex(Is)
    I = Is[n]
    θ = θs[I[1], I[2]]
    η = η_func(θ, ωs, L, ωs)
    J = η_func_jac(θ, ωs, L, ωs)
    fisher_values_obs[n] = tr(inv(J' * fisher(mthd, η) * J))
    next!(p)
end

modes = [[abs2(f(x, y)) for x ∈ rs, y ∈ rs] for f in basis_func_obs]
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
        title = L"x_b = %$xb w")
    xlims!(ax, (-1, 1))
    ylims!(ax, (-1, 1))


    log_relative = log2.(fisher_values_obs ./ fisher_values)
    colormap = get_colormap(log_relative)

    lim = maximum(abs, log_relative)
    hm = heatmap!(ga[1,1], x_coords, y_coords, log_relative; colormap)
    Colorbar(ga[1,2], hm, label=L"\log_2 \left( \text{Tr } I^{-1} / \text{Tr } I_0^{-1} \right)")

    ax2 = Axis(gb[1, 1], aspect=DataAspect())
    hidedecorations!(ax2)
    hm2 = heatmap!(ax2, modes; colormap=:jet)
    Colorbar(gb[1, 2], hm2, label="Intensity")

    rowsize!(g, 2, Auto(0.5))

    #save("Plots/fisher_sphere_xb=$xb.png", fig, px_per_unit=4)

    fig
end