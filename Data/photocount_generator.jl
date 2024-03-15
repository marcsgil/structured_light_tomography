using ProgressMeter

include("dataset_generation.jl")

file = h5open("Data/Training/pure_photocount.h5", "cw")

orders = 1:5
photocounts = [2^k for k ∈ 6:11]
p = Progress(length(orders) * length(photocounts))

for order ∈ orders
    images, labels = nothing, nothing
    GC.gc()
    R = 2.5f0 + 0.5f0 * order
    rs = LinRange(-R, R, 64)
    for pc ∈ photocounts
        images, labels = generate_dataset(order, 10^5, rs, π / 2, PureState())
        add_noise!(images)
        sample_photons!(images, pc)

        file["images_order$(order)/$(pc)_photocounts"] = images
        file["labels_order$(order)/$(pc)_photocounts"] = labels
        next!(p)
    end
end
finish!(p)
close(file)