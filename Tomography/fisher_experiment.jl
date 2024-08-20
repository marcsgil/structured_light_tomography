using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography, HDF5
includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("../Utils/obstructed_measurements.jl")

function cov_bound(θ, mthd, L, ωs)
    η = η_func(θ, ωs, L, ωs)
    J = η_func_jac(θ, ωs, L, ωs)
    sum(inv, eigvals(J' * fisher(mthd, η) * J))
end

function average_cov_bound(θs, rs, ωs, basis_func, obstruction_func, args...; kwargs...)
    basis_func_obs = get_obstructed_basis(basis_func, obstruction_func, args...; kwargs...)
    T, Ω, L = assemble_povm_matrix(rs, rs, basis_func_obs)
    mthd = LinearInversion(T, Ω)

    mean(cov_bound(θ, mthd, L, ωs) for θ ∈ eachslice(θs, dims=2))
end

extract_θ(ρ, ωs) = stack(real(tr(ρ * ω)) for ω ∈ eachslice(view(ωs, :, :, 2:4), dims=3))
##
rs = LinRange(-2.2f0, 2.2f0, 256)
basis_func = positive_l_basis(2, [0.0f0, 0, 1, 1])
ωs = gell_mann_matrices(2)

ρs = h5open("Data/template.h5") do file
    read(file, "labels_dim2")
end

θs = stack(extract_θ(ρ, ωs) for ρ ∈ eachslice(ρs, dims=3))
##
radius = LinRange(0.4f0, 2f0, 200)
bounds = [average_cov_bound(θs, rs, ωs, basis_func, iris_obstruction, 0, 0, radius) for radius in radius]
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1], 
    xlabel = "Radius", 
    ylabel = "Average MSE bound",
    yscale = log2,
    xticks = 0.4:0.4:2,
    yticks = [2^k for k ∈ 2:1:12]
    )
    lines!(ax, radius, bounds, linewidth = 4)
    #save("Plots/iris_bound.pdf", fig)
    fig
end
##
xb = LinRange(-1.5, 2, 200)
bounds = [average_cov_bound(θs, rs, ωs, basis_func, blade_obstruction, xb) for xb in xb]
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1], 
    xlabel = "Blade position", 
    ylabel = "Average MSE bound",
    yscale = log2,
    xticks = -1.5:0.5:2,
    yticks = [2^k for k ∈ 2:1:12]
    )
    lines!(ax, xb, bounds, linewidth = 4)
    #save("Plots/blade_bound.pdf", fig)
    fig
end