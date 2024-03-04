using TiffImages, ProgressMeter, HDF5, BayesianTomography

include("data_treatment_utils.jl")

file = h5open("Data/Raw/mixed_intense.h5")
out = h5open("Data/Processed/mixed_intense.h5", "cw")

calibration = read(file["calibration"])

remove_background!(calibration)
direct_lims, converted_lims = get_limits(calibration)

out["direct_lims"] = direct_lims
out["converted_lims"] = converted_lims

out["weights"] = [sum(calibration[:, :, 1]), sum(calibration[:, :, 2])] / sum(calibration)

for order ∈ 1:5
    images = Float64.(read(file["images_order$order"]))
    remove_background!(images)

    for image ∈ eachslice(images, dims=(3, 4))
        normalize!(image, 1)
        @. image /= 2
    end
    out["images_order$order"] = images
    HDF5.attributes(out["images_order$order"])["angle"] = read_attribute(file["images_order$order"], "angle")

    #Python and Julia have different conventions for saving arrays
    out["labels_order$order"] = permutedims(read(file["labels_order$order"]), (2, 1, 3))
end

sum(read(out["images_order4"])[:, :, :, 1])

close(out)
close(file)