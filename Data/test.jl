using Distributions, LinearAlgebra, CairoMakie, BayesianTomography, ProgressMeter, HDF5
includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("dataset_generation.jl")

function calculate_mode!(dest, ρ, dist_waist, dist_center)
    order = size(ρ, 1) - 1
    pars_d = (rand(dist_center), rand(dist_center), rand(dist_waist))
    pars_a = (rand(dist_center), rand(dist_center), rand(dist_waist))

    basis_func_direct = fixed_order_basis(order, pars_d)
    basis_func_astig = [(x, y) -> f(x, y) * cis(-(k - 1) * Float32(π) / 6)
                        for (k, f) ∈ enumerate(fixed_order_basis(order, pars_a))]
    buffer = [f(0.0f0, 0.0f0) for f in basis_func_direct]
    get_intensity!(view(dest, :, :, 1), buffer, ρ, basis_func_direct, rs, rs)
    get_intensity!(view(dest, :, :, 2), buffer, ρ, basis_func_astig, rs, rs)
end
##
d_waist = LogNormal(-1.7f0, 0.2f0)

ws = LinRange(0, 1, 512)
probs = pdf.(d_waist, ws)

μ = round(mean(d_waist), digits=2)
_mode = round(mode(d_waist), digits=2)

fig, ax, = lines(ws, probs)
ax.title = "Mode: $_mode; Mean: $μ"
fig
##
d_center = Normal(0.0f0, 0.05f0)

r = LinRange(-0.5, 0.5, 512)
probs = pdf.(d_center, r)

μ = round(mean(d_center), digits=2)
_mode = round(mode(d_center), digits=2)

fig, ax, = lines(r, probs)
ax.title = "Mode: $_mode; Mean: $μ"
fig


rand(Float32, d_waist)
##
rs = LinRange(-0.5f0, 0.5f0, 64)
dest = Array{Float32}(undef, length(rs), length(rs), 2, 10^5)
ρs = BayesianTomography.sample(GinibreEnsamble(6), size(dest, 4))
θs = stack(gell_mann_projection, eachslice(ρs, dims=3))
##
p = Progress(size(dest, 4))
Threads.@threads for n ∈ axes(dest, 4)
    calculate_mode!(view(dest, :, :, :, n), view(ρs, :, :, n), d_waist, d_center)
    next!(p)
end
finish!(p)

add_noise!(dest, dims=(3,4))
##
visualize(dest[:, :, 1, 17])
##

h5open("Data/Training/fixed_order_intense.h5", "cw") do file
    file["images_order5"] = dest
    file["labels_order5"] = θs
end
