using BayesianTomography, HDF5, CairoMakie, LinearAlgebra

function history2dict(history)
    result = Dict{Int,Int}()
    for event ∈ history
        result[event] = get(result, event, 0) + 1
    end
    result
end

file = h5open("Data/ExperimentalData/Old/ExperimentalResults/UFMG/config.h5")
direct_limits = read(file["direct_limits"])
converted_limits = read(file["converted_limits"])
close(file)

L = 64
direct_x = LinRange(direct_limits[1], direct_limits[3], L)
direct_y = LinRange(direct_limits[2], direct_limits[4], L)
converted_x = LinRange(converted_limits[1], converted_limits[3], L)
converted_y = LinRange(converted_limits[2], converted_limits[4], L)

photocounts = [2^k for k ∈ 6:11]

file = h5open("Results/Photocount/bayes.h5")
fids = file["fids"] |> read
fids_std = file["fids_std"] |> read
close(file)
##
order = 3
file = h5open("Data/ExperimentalData/Photocount/datasets.h5")
histories = read(file["histories_order$order"])
coefficients = read(file["coefficients_order$order"])
close(file)

direct_operators = assemble_position_operators(direct_x, direct_y, order)
mode_converter = diagm([cis(-k * π / 2) for k ∈ 0:order])
astig_operators = assemble_position_operators(converted_x, converted_y, order)
astig_operators = unitary_transform(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)

method = MetropolisHastings()

for n ∈ eachindex(photocounts)
    photocount = photocounts[n]
    this_fids = Vector{Float64}(undef, size(histories, 2))
    Threads.@threads for m ∈ axes(histories, 2)
        outcomes = history2dict(view(histories, 1:photocount, m))
        predicted_c = prediction(outcomes, operators, method) |> hurwitz_parametrization
        this_fids[m] = abs2(coefficients[:, m] ⋅ predicted_c)
    end
    fids[n, order] = mean(this_fids)
    fids_std[n, order] = std(this_fids)
end

fids
fids_std

lines(log2.(photocounts), fids[:, order], color=:blue, label="Machine learning")
##
file = h5open("Results/Photocount/bayes.h5", "cw")
file["fids"] = fids
file["fids_std"] = fids_std
close(file)