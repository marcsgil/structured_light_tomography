using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, QuantumMeasurements, HDF5
includet("../../Utils/basis.jl")
includet("../../Utils/obstructions.jl")

function average_cov_bound(θs, rs, itr, obstruction_func, args...; kwargs...)
    I = get_valid_indices(rs, rs, obstruction_func, args...; kwargs...)
    μ = ProportionalMeasurement(itr[I])

    mean(sum(inv, eigvals(fisher(μ, θ))) for θ ∈ eachslice(θs, dims=2))
end
##
rs = LinRange(-2.2f0, 2.2f0, 256)
sqrt_δA = rs[2] - rs[1]
itr = [positive_l_basis(2, (x, y), (sqrt_δA, 0, 0, 1)) for x ∈ rs, y ∈ rs]

ρs = h5open("Data/template.h5") do file
    read(file, "labels_dim2")
end

μ = assemble_measurement_matrix(itr)

sum(inv, eigvals(fisher(μ, [0, 0, 0])))

θs = stack(traceless_vectorization(ρ) for ρ ∈ eachslice(ρs, dims=3))
##
radii = LinRange(0.4f0, 2.0f0, 50)
bounds = Vector{Float32}(undef, length(radii))

p = Progress(length(radii))
Threads.@threads for n ∈ eachindex(bounds)
    r = radii[n]
    bounds[n] = average_cov_bound(θs, rs, itr, iris_obstruction, 0, 0, r)
    next!(p)
end
finish!(p)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1],
        xlabel=L"r \ (w )",
        ylabel=L"B",
        yscale=log2,
        xticks=0.4:0.4:2,
        yticks=[2^k for k ∈ 2:1:12]
    )
    lines!(ax, radii, bounds, linewidth=4)
    #save("Plots/iris_bound.pdf", fig)
    fig
end
##
xb = LinRange(-1.5, 2, 200)
bounds = Vector{Float32}(undef, length(xb))

p = Progress(length(xb))
Threads.@threads for n ∈ eachindex(bounds)
    bounds[n] = average_cov_bound(θs, rs, itr, blade_obstruction, xb[n])
    next!(p)
end
finish!(p)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1],
        xlabel=L"x_b \ (w)",
        ylabel=L"B",
        yscale=log2,
        xticks=-1.5:0.5:2,
        yticks=[2^k for k ∈ 2:1:12]
    )
    lines!(ax, xb, bounds, linewidth=4)
    #save("Plots/blade_bound.pdf", fig)
    fig
end