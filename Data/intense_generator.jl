using ProgressMeter

include("dataset_generation.jl")

file = h5open("Data/Training/mixed_intense.h5", "cw")
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
close(file)