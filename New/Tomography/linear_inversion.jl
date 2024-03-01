using HDF5, CairoMakie, StatsBase, BayesianTomography, ProgressMeter
includet("../Data/fit_grid.jl")
includet("../utils.jl")

most_frequent_value(data) = countmap(data) |> argmax

function remove_background!(images)
    bg = most_frequent_value(images)
    map!(x -> x < bg ? zero(x) : x - bg, images, images)
end

input = h5open("New/Data/Intense/mixed.h5")
calibration = read(input["calibration"])

remove_background!(calibration)
direct_lims, converted_lims = get_limits(calibration)
xd, yd = get_grid(direct_lims, size(calibration))
xc, yc = get_grid(converted_lims, size(calibration))

order = 1
images = read(input["images_order$order"])
ρs = read(input["labels_order$order"])
basis = [(r, par) -> hg(r[1], r[2], m, order - m) for m ∈ 0:order]
##

##
direct_operators = assemble_position_operators(xd, yd, basis)
converted_operators = assemble_position_operators(xc, yc, basis)
mode_converter = diagm([cis(-k * π / 6) for k ∈ 0:order])
unitary_transform!(converted_operators, mode_converter)
operators = compose_povm(direct_operators, converted_operators)
fids = Vector{Float64}(undef, size(images, 4))
mthd = LinearInversion(operators)

p = Progress(length(fids));
Threads.@threads for n ∈ eachindex(fids)
    direct = normalize(images[:, :, 1, n], 1)
    converted = normalize(images[:, :, 2, n], 1)
    probs = cat(direct / 2, converted / 2, dims=3)
    σ = project2density(prediction(probs, mthd))
    fids[n] = fidelity(ρs[:, :, n], σ)
    next!(p)
end
finish!(p)

fids
mean(fids)
