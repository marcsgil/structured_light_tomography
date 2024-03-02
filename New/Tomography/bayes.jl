using BayesianTomography, HDF5

includet("../utils.jl")
includet("../Data/data_treatment_utils.jl")

file = h5open("New/Data/Photocount/data.h5")

calibration = read(file["calibration"])
remove_background!(calibration)
direct_lims, converted_lims = get_limits(calibration)
direct_x, direct_y = get_grid(direct_lims, (64, 64))
converted_x, converted_y = get_grid(converted_lims, (64, 64))

out = h5open("New/Results/Photocount/bayes.h5")
fids = read(out["fids"])
close(out)
##
mtdh = MetropolisHastings()
photocounts = [2^k for k ∈ 6:11]
orders = 3

for order ∈ orders
    @show order

    histories = file["histories_order$order"] |> read
    coefficients = read(file["coefficients_order$order"])

    basis = [(r, par) -> hg(r[1], r[2], m, order - m) for m ∈ 0:order]

    direct_operators = assemble_position_operators(direct_x, direct_y, basis)
    mode_converter = diagm([cis(k * π / 2) for k ∈ 0:order])
    astig_operators = assemble_position_operators(converted_x, converted_y, basis)
    unitary_transform!(astig_operators, mode_converter)
    operators = compose_povm(direct_operators, astig_operators)

    for n ∈ 2#eachindex(photocounts)
        photocount = photocounts[n]
        this_fids = Vector{Float64}(undef, size(histories, 2))
        Threads.@threads for m ∈ axes(histories, 2)
            outcomes = history2dict(view(histories, 1:photocount, m))
            predicted_c = prediction(outcomes, operators, mtdh) |> hurwitz_parametrization
            this_fids[m] = abs2(coefficients[:, m] ⋅ predicted_c)
        end
        fids[n, order] = mean(this_fids)
    end
end

fids
##
out = h5open("New/Results/Photocount/bayes.h5", "w")
out["fids"] = fids
out["photocounts"] = photocounts
close(out)