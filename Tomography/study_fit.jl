using HDF5, StatsBase, StructuredLight
includet("../Utils/model_fitting.jl")

most_frequent_value(data) = countmap(data) |> argmax

function remove_background!(images)
    bg = most_frequent_value(images)
    bg += one(bg)
    map!(x -> x < bg ? zero(x) : x - bg, images, images)
end
##
calibration = h5open("Data/Raw/test_calib.h5") do file
    file["calibration"] |> read
end .|> Float32

remove_background!(calibration)

rs = LinRange(-0.5, 0.5, size(calibration, 1))

calibration ./= sum(calibration) * (rs[2] - rs[1])^2

function simple_model(x, y, p)
    x₀, y₀, w = p
    exp(-2 * ((x - x₀)^2 + (y - y₀)^2) / w^2) * 2 / (π * w^2)
end

p0 = [0.0, 0.0, 0.1]

fit = surface_fit(simple_model, rs, rs, calibration, p0)

fit.param

cp_fit_param = copy(fit.param)
cp_fit_param[3] /= 1.1

se(x,y) = mapreduce((x, y) -> abs2(x - y), +, x, y)

se([simple_model(x, y, fit.param) for x ∈ rs, y ∈ rs], calibration)