using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography, HDF5
includet("../../Utils/basis.jl")
includet("../../Utils/obstructions.jl")
includet("../../Utils/position_operators.jl")

function average_cov_bound(θs, rs, measurement, obstruction_func, args...; kwargs...)
    I = get_valid_indices(rs, rs, obstruction_func, args...; kwargs...)
    obstructed_measurement = ProportionalMeasurement(measurement[I])

    mean(sum(inv, eigvals(fisher(obstructed_measurement, θ))) for θ ∈ eachslice(θs, dims=2))
end
##
rs = LinRange(-2.2f0, 2.2f0, 256)
basis_func = positive_l_basis(2, [0.0f0, 0, 1, 1])
measurement = assemble_position_operators(rs, rs, basis_func)

ρs = h5open("Data/template.h5") do file
    read(file, "labels_dim2")
end

θs = stack(gell_mann_projection(ρ) for ρ ∈ eachslice(ρs, dims=3))
##
radius = LinRange(0.4f0, 2.0f0, 200)
bounds = [average_cov_bound(θs, rs, measurement, iris_obstruction, 0, 0, radius) for radius in radius]
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1],
        xlabel="Radius",
        ylabel="Average MSE bound",
        yscale=log2,
        xticks=0.4:0.4:2,
        yticks=[2^k for k ∈ 2:1:12]
    )
    lines!(ax, radius, bounds, linewidth=4)
    #save("Plots/iris_bound.pdf", fig)
    fig
end
##
xb = LinRange(-1.5, 2, 200)
bounds = [average_cov_bound(θs, rs, measurement, blade_obstruction, xb) for xb in xb]
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1],
        xlabel="Blade position",
        ylabel="Average MSE bound",
        yscale=log2,
        xticks=-1.5:0.5:2,
        yticks=[2^k for k ∈ 2:1:12]
    )
    lines!(ax, xb, bounds, linewidth=4)
    #save("Plots/blade_bound.pdf", fig)
    fig
end