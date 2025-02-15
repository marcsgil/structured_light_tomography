using LinearAlgebra, QuantumMeasurements, CairoMakie
includet("../Utils/basis.jl")

N = 512
x = LinRange(-6, 6, N)
rs = Iterators.product(x, x)
##
heatmap(abs2.(hg(x, x, m=29)))
##
pars = ((x[2] - x[1]), 0.0, 0.0, 1.0)
dimensions = 2:30
norm_rank = Vector{Float64}(undef, length(dimensions))

Threads.@threads for n ∈ eachindex(dimensions, norm_rank)
    d = dimensions[n]

    itr = Iterators.map(r -> fixed_order_basis(d - 1, r, pars), rs)
    μ = assemble_measurement_matrix(itr)
    T = get_traceless_part(μ)

    norm_rank[n] = rank(T)
end

with_theme(theme_latexfonts()) do 
    fig = Figure(size = (600, 300), fontsize = 24)
    ax = Axis(fig[1, 1]; xlabel = "Dimension", xticks = 0:5:30)
    lines!(ax, dimensions, (dimensions) .* (dimensions .+ 1) ./ 2 .- 1, color = :red, linewidth = 5, label = L"y = d(d + 1) / 2 - 1")
    lines!(ax, dimensions, dimensions.^2 .- 1, color = :blue, linewidth = 5, label = L"y = d^2 - 1")
    scatter!(ax, dimensions, norm_rank, color = :black, markersize = 14, marker = :cross, label = "Rank of T")
    axislegend(position = :lt)
    save("../Structured light tomography from position measurements/no_lens_rank.pdf", fig)
    fig
end 
##
pars = ((x[2] - x[1]) / 2, 0.0, 0.0, 1.0)
dimensions = 2:30
factors = Vector(5:5:30)
angles = π ./ factors
norm_rank = Matrix{Float64}(undef, length(dimensions), length(angles))

Threads.@threads for n ∈ eachindex(dimensions)
    d = dimensions[n]

    itr1 = Iterators.map(r -> fixed_order_basis(d - 1, r, pars), rs)
    μ1 = assemble_measurement_matrix(itr1)

    for m ∈ eachindex(angles)
        itr2 = Iterators.map(r -> fixed_order_basis(d - 1, r, pars, angles[m]), rs)
        μ2 = assemble_measurement_matrix(itr2)
        μ = vcat(μ1, μ2)
        T = get_traceless_part(μ)

        norm_rank[n, m] = rank(T)
    end
end

with_theme(theme_latexfonts()) do 
    fig = Figure(size = (800, 400), fontsize = 28)
    ax = Axis(fig[1, 1]; xlabel = "Dimension", ylabel = L"\text{rank}(T) / (d^2 - 1)", xticks = 0:5:30, yscale = identity)
    colors = (:red, :green, :blue, :purple, :orange, :black)
    markers = [:circle, :cross, :diamond, :rect, :xcross, :hexagon]
    for m ∈ reverse(eachindex(factors))
        scatter!(ax, dimensions, norm_rank[:, m] ./ (dimensions.^2 .- 1), color = colors[m], markersize = 20, marker = markers[m], label = L"θ = \pi / %$(factors[m])")
    end
    #axislegend(position = :lb)
    Legend(fig[1, 2], ax)
    save("../Structured light tomography from position measurements/lens_rank.pdf", fig)
    fig
end 