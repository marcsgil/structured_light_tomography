using Tullio, BayesianTomography, CairoMakie, HDF5, Distributions

include("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("dataset_generation.jl")
##
d_waist = LogNormal(-2.0f0, 0.3f0)

ws = LinRange(0, 0.5, 512)
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
##
rs = LinRange(-0.5f0, 0.5f0, 64)
x = Array{Float32}(undef, length(rs), length(rs), 1, 4 * 10^5)
y = vcat(rand(d_center, 2, size(x, 4)), Float32.(rand(d_waist, 1, size(x, 4))))

p = Progress(size(x, 4))

Threads.@threads for n ∈ axes(x, 4)
    order = n ÷ (size(x, 4) ÷ 10) + 1
    basis_func = fixed_order_basis(order, view(y, :, n))
    buffer = [f(rs[1], rs[1]) for f in basis_func]

    ρ = BayesianTomography.sample(ProductMeasure(order + 1))

    get_intensity!(view(x, :, :, 1, n), buffer, ρ, basis_func, rs, rs)
    next!(p)
end

finish!(p)
##
visualize(x[:, :, 1, 4*10^5])
##
GC.gc()

add_noise!(x, dims=(3, 4))
##


h5open("Data/Training/center_and_waist.h5", "cw") do file
    file["x"] = x
    file["y"] = y
end