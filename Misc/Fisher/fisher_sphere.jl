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

function get_data(x, y, θx, θz, operators, basis_func, obstruction_func, args...; kwargs...)
    @assert length(θx) == length(θz)

    I = get_valid_indices(x, y, obstruction_func, args...; kwargs...)
    measurement = ProportionalMeasurement(operators[I])

    bound_values = Vector{Float32}(undef, length(θx))
    p = Progress(length(bound_values))
    Threads.@threads for n ∈ eachindex(bound_values)
        @inbounds θ = [θx[n], 0, θz[n]]
        bound_values[n] = sum(inv, eigvals(fisher(measurement, θ)))
        next!(p)
    end

    modes = [[abs2(f(x, y)) for x ∈ x, y ∈ y] for f in get_obstructed_basis(basis_func, obstruction_func, args...; kwargs...)]
    for mode ∈ modes
        normalize!(mode, Inf)
    end
    modes = vcat(modes[1], modes[2])
    bound_values, modes
end

function make_plot(data, θx, θz, colorbar_label, titles=["" for _ ∈ data]; saving_path="", reference=nothing)
    with_theme(theme_latexfonts()) do
        fig = Figure(size=(length(data) * 600, 700), fontsize=24)

        for (n, (bounds, modes)) ∈ enumerate(data)
            ax = Axis(fig[1, 2n-1],
                aspect=DataAspect(),
                xlabel=L"x",
                ylabel=L"z",
                xlabelsize=32,
                ylabelsize=32,
                xgridvisible=false,
                ygridvisible=false,
                title=titles[n])
            xlims!(ax, (-1, 1))
            ylims!(ax, (-1, 1))

            if !isnothing(reference)
                log_relative = @. log2(bounds / reference)
                colormap = get_colormap(log_relative)
                hm = heatmap!(ax, θx * √2, θz * √2, log_relative; colormap)
            else
                hm = heatmap!(ax, θx * √2, θz * √2, bounds)
            end

            Colorbar(fig[1, 2n], hm, label=colorbar_label)

            ax2 = Axis(fig[2, 2n-1], aspect=DataAspect())
            hidedecorations!(ax2)

            hm2 = heatmap!(ax2, modes; colormap=:jet)
            Colorbar(fig[2, 2n], hm2, label="Intensity")
        end

        rowsize!(fig.layout, 2, Auto(0.5))

        isempty(saving_path) || save(saving_path, fig, px_per_unit=4)

        fig
    end
end
##
rs = LinRange(-2.2f0, 2.2f0, 256)
basis_func = positive_l_basis(2, (0.0f0, 0, 1))

operators = assemble_position_operators(rs, rs, basis_func)

measurement = Measurement(operators)
##
coords = LinRange(-1 / Float32(√2), 1 / Float32(√2), 512)
θs = [[x, 0, z] for x in coords, z ∈ coords]

Is = findall(θ -> sum(abs2, θ) ≤ 1 / 2, θs)

θx = [θs[I][1] for I ∈ Is]
θz = [θs[I][3] for I ∈ Is]


unobstructed_bounds, unobstructed_modes = get_data(rs, rs, θx, θz, operators, basis_func, (x, y) -> true)
##
make_plot([(unobstructed_bounds, unobstructed_modes)], θx, θz, L"\text{Tr } I^{-1}")
##
radiuses = (1.0f0, 0.5f0, 0.25f0)
iris_data = [get_data(rs, rs, θx, θz, operators, basis_func, iris_obstruction, 0, 0, r) for r ∈ radiuses]
titles = ["Radius = $r w" for r ∈ radiuses]

colorbar_label = L"\log_2 \left( \text{Tr } I^{-1} / \text{Tr } I_0^{-1} \right)"

make_plot(iris_data, θx, θz, colorbar_label, titles, reference=unobstructed_bounds)
##
xbs = (-0.5, -1, -1.5)
blade_data = [get_data(rs, rs, θx, θz, operators, basis_func, blade_obstruction, xb) for xb ∈ xbs]
titles = ["Blade position = $xb w" for xb ∈ xbs]

colorbar_label = L"\log_2 \left( \text{Tr } I^{-1} / \text{Tr } I_0^{-1} \right)"

make_plot(blade_data, θx, θz, colorbar_label, titles, reference=unobstructed_bounds)