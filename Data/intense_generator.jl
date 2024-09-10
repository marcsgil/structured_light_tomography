using ProgressMeter, BayesianTomography

includet("dataset_generation.jl")
include("../Utils/basis.jl")

R = 3.0f0
rs = LinRange(-R, R, 64)
basis_func = fixed_order_basis(1, [0, 0, sqrt(2), 1])
basis_direct = stack(f(x, y) for x in rs, y in rs, f ∈ basis_func)
basis_astig = stack(f(x, y) * cis(-(k - 1) * π / 6) for x in rs, y in rs, (k, f) ∈ enumerate(basis_func))

ρs = BayesianTomography.sample(ProductMeasure(length(basis_func)), 10^5)
θs = stack(gell_mann_projection(ρ) for ρ in eachslice(ρs, dims=3))
dest = Array{Float32}(undef, length(rs), length(rs), 2, size(θs, 2))

generate_dataset!(view(dest, :, :, 1, :), basis_direct, ρs)
generate_dataset!(view(dest, :, :, 2, :), basis_astig, ρs)
add_noise!(view(dest, :, :, 1, :))
add_noise!(view(dest, :, :, 2, :))

h5open("Data/Training/mixed_intense.h5", "cw") do file
    file["images_order1"] = dest
    file["labels_order1"] = θs
end
##

images, labels = h5open("Data/Processed/mixed_intense.h5") do file
    read(file["images_order1"]), read(file["labels_order1"])
end

images, labels = h5open("Data/Raw/positive_l.h5") do file
    obj = file["images_dim2"]
    read(obj), attrs(obj)["density_matrices"]
end


dest_test = Array{Float32}(undef, 64, 64, 100)


basis_func = positive_l_basis(2, [0, 0, sqrt(2), 1])
basis = stack(f(x, y) for x in rs, y in rs, f ∈ basis_func)
generate_dataset!(dest_test, basis_astig, labels)


visualize(dest_test[:, :, 4])
visualize(images[:, :, 2, 4])