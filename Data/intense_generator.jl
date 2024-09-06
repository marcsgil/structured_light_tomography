using ProgressMeter, BayesianTomography

includet("dataset_generation.jl")
include("../Utils/basis.jl")

R = 3f0
rs = LinRange(-R, R, 64)
basis_func = positive_l_basis(2, [0,0,1f0,1])
basis = stack(f(x,y) for x in rs, y in rs, f ∈ basis_func)

ρs = BayesianTomography.sample(ProductMeasure(length(basis_func)), 10^5)
θs = stack(gell_mann_projection(ρ) for ρ in eachslice(ρs, dims=3))
dest = Array{Float32}(undef, length(rs), length(rs), size(θs, 2))

generate_dataset!(dest, basis, ρs)
#add_noise!(dest)

h5open("Data/Training/positive_l.h5", "cw") do file
    file["images"] = dest
    file["labels"] = θs
end
##

imgs, ρs = h5open("Data/Raw/positive_l.h5") do file
    obj = file["images_dim2"]
    read(obj), attrs(obj)["density_matrices"]
end

