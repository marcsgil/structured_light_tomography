using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography
includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("../Utils/obstructed_measurements.jl")
##
rs = LinRange(-3.0f0, 3, 256)
basis_func = positive_l_basis(2, [0.0f0, 0, 1, 1])
##
T, Ω, L = assemble_povm_matrix(rs, rs, basis_func)
mthd = LinearInversion(T, Ω)

coords = LinRange(-1.0f0, 1, 200)
inv_sqrt_2 = Float32(1 / √2)
θs = [[x * inv_sqrt_2, 0, z * inv_sqrt_2] for x in coords, z ∈ coords]

Is = findall(θ -> sum(abs2, θ) ≤ 1 / 2, θs)

x_coords = [coords[I[1]] for I ∈ Is]
y_coords = [coords[I[2]] for I ∈ Is]

fisher_values = Vector{Float32}(undef, length(Is))

Threads.@threads for n ∈ eachindex(Is)
    I = Is[n]
    fisher_values[n] = tr(inv(fisher(mthd, θs[I[1], I[2]])))
end
##
fig = Figure(size=(700, 600))
ax = Axis(fig[1, 1], aspect=DataAspect())
xlims!(ax, (-1, 1))
ylims!(ax, (-1, 1))
hm = heatmap!(ax, x_coords, y_coords, fisher_values)
Colorbar(fig[1, 2], hm)
fig
##
ωs = gell_mann_matrices(2)
basis_func_obs = [(x, y) -> f(x, y) * blade_obstruction(x, y, -1) for f in basis_func]
T, Ω, L = assemble_povm_matrix(rs, rs, basis_func_obs)
mthd = LinearInversion(T, Ω)

fisher_values_obs = Vector{Float32}(undef, length(Is))

Threads.@threads for n ∈ eachindex(Is)
    I = Is[n]
    θ = θs[I[1], I[2]]
    J = η_func_jac(θ, ωs, L, ωs)
    fisher_values_obs[n] = tr(inv(J' * fisher(mthd, θ) * J))
end

fig = Figure(size=(700, 600))
ax = Axis(fig[1, 1], aspect=DataAspect())
xlims!(ax, (-1, 1))
ylims!(ax, (-1, 1))

log_relative = log.(fisher_values_obs ./ fisher_values)

lim = maximum(abs, log_relative)
hm = heatmap!(ax, x_coords, y_coords, log_relative, colormap=:seismic, colorrange=(-lim, lim))   
Colorbar(fig[1, 2], hm)
fig