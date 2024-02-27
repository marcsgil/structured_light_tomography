using HDF5, CairoMakie, StatsBase

most_frequent_value(data) = countmap(data) |> argmax

relu(x, y) = x < y ? zero(x) : x - y

input = h5open("New/Data/Intense/Raw/mixed.h5")
#output = h5open("New/Data/Intense/mixed.h5", "cw")

images = read(input["images_order3"])
minimum(images)

most_frequent_value(images)
@benchmark most_frequent_value($images)


heatmap(images[:, :, 1, 1])


calibration = read(input["calibration"])

bg = most_frequent_value(calibration[:, :, 1])
bg += one(bg)