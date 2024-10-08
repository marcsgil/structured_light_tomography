using ProgressMeter, BayesianTomography, HDF5

includet("dataset_generation.jl")
include("../Utils/basis.jl")
##
dest = Array{Float32}(undef, length(x), length(y), 2, 10^5)

r = LinRange(-3.,3.,64)

@showprogress for order ∈ 5:5
    basis_func = fixed_order_basis(order, (0,0,1f0))
    basis_direct = stack(f(x, y) for x in r, y in r, f ∈ basis_func)
    basis_func = fixed_order_basis(order, (0,0,1f0))
    basis_astig = stack(f(x, y) * cis(-(k - 1) * Float32(π) / 6) for x in r, y in r, (k, f) ∈ enumerate(basis_func))

    ρs = BayesianTomography.sample(ProductMeasure(length(basis_func)), size(dest, 4))
    θs = stack(gell_mann_projection(ρ) for ρ in eachslice(ρs, dims=3))

    generate_dataset!(view(dest, :, :, 1, :), basis_direct, ρs)
    generate_dataset!(view(dest, :, :, 2, :), basis_astig, ρs)
    add_noise!(view(dest, :, :, 1, :))
    add_noise!(view(dest, :, :, 2, :))

    h5open("Data/Training/not_fitted_fixed_order_intense.h5", "cw") do file
        file["images_order$order"] = dest
        file["labels_order$order"] = θs
    end
end