using ProgressMeter, BayesianTomography

includet("dataset_generation.jl")
include("../Utils/basis.jl")

"""file = h5open("Data/Training/mixed_intense.h5", "cw")
@showprogress for order ∈ 1:5
    images, labels = nothing, nothing
    GC.gc()
    R = 2.5f0 + 0.5f0 * order
    rs = LinRange(-R, R, 64)
    images, labels = generate_dataset(order, 10^5, rs, -π / 6, MixedState())
    add_noise!(images)

    file["images_order$(order)"] = images
    file["labels_order$(order)"] = labels
end
close(file)"""

R = 3f0
rs = LinRange(-R, R, 64)
basis_func = positive_l_basis(2, [0,0,1f0,1])
basis = stack(f(x,y) for x in rs, y in rs, f ∈ basis_func)

ρs = BayesianTomography.sample(ProductMeasure(length(basis_func)), 10^5)
θs = stack(gell_mann_projection(ρ) for ρ in eachslice(ρs, dims=3))
dest = Array{Float32}(undef, length(rs), length(rs), size(θs, 2))

generate_dataset!(dest, basis, ρs)
add_noise!(dest)

visualize(dest[:,:,1])

h5open("Data/Training/positive_l.h5", "cw") do file
    file["images"] = dest
    file["labels"] = θs
end
