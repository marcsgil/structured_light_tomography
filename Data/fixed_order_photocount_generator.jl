using ProgressMeter, BayesianTomography, LinearAlgebra, HDF5

includet("dataset_generation.jl")
include("../Utils/basis.jl")
include("../Utils/pure_state_utils.jl")

dest = Array{Float32}(undef, 64, 64, 2, 10^5)

orders = 1:4
photocounts = [2^k for k ∈ 6:11]
p = Progress(length(orders) * length(photocounts))

for order ∈ orders
    R = 2.5f0 + 0.5f0 * order
    rs = LinRange(-R, R, 64)

    basis_func = fixed_order_basis(order, [0, 0, √2.0f0, 1])
    basis_direct = stack(f(x, y) for x in rs, y in rs, f ∈ basis_func)
    basis_astig = stack(f(x, y) * cis((k - 1) * Float32(π) / 2) for x in rs, y in rs, (k, f) ∈ enumerate(basis_func))

    for pc ∈ photocounts
        ψs = BayesianTomography.sample(HaarVector(order+1), size(dest, 4))
        generate_dataset!(view(dest, :, :, 1, :), basis_direct, ψs)
        generate_dataset!(view(dest, :, :, 2, :), basis_astig, ψs)


        Threads.@threads for slice ∈ eachslice(dest, dims=4)
            normalize!(slice, 1)
            simulate_outcomes!(slice, pc)
        end

        h5open("Data/Training/fixed_order_photocount.h5", "cw") do file
            file["images_order$(order)/$(pc)_photocounts"] = dest
            file["labels_order$(order)/$(pc)_photocounts"] = cat_real_and_imag(ψs)
        end

        next!(p)
    end
end
finish!(p)