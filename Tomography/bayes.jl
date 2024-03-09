using BayesianTomography, HDF5, ProgressMeter

includet("../utils.jl")
includet("../Data/data_treatment_utils.jl")

file = h5open("Data/Processed/pure_photocount.h5")

direct_lims = read(file["direct_lims"])
converted_lims = read(file["converted_lims"])
direct_x, direct_y = get_grid(direct_lims, (64, 64))
converted_x, converted_y = get_grid(converted_lims, (64, 64))

out = h5open("Results/Photocount/bayes.h5")
fids = read(out["fids"])
close(out)
##
order = 1
histories = file["histories_order$order"] |> read
coefficients = read(file["labels_order$order"])

basis = transverse_basis(order) |> reverse

direct_operators = assemble_position_operators(direct_x, direct_y, basis)
mode_converter = diagm([cis(k * π / 2) for k ∈ 0:order])
astig_operators = assemble_position_operators(converted_x, converted_y, basis)
unitary_transform!(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)
##
mtdh = BayesianInference(operators, 10^6, 10^3)
hermitian_basis = get_hermitian_basis(order + 1)
photocount = 2048
m = 1

outcomes = history2dict(view(histories, 1:photocount, m))
xs = prediction(outcomes, mtdh) |> mean
ρ = linear_combination(xs, hermitian_basis)
ψ = project2pure(ρ)


abs2(coefficients[:, m] ⋅ ψ)
##
orders = 1:4
photocounts = [2^k for k ∈ 6:11]
all_fids = zeros(Float64, length(photocounts), 50, length(orders))

for k ∈ eachindex(orders)
    order = orders[k]
    histories = file["histories_order$order"] |> read
    coefficients = read(file["labels_order$order"])

    basis = transverse_basis(order) |> reverse

    direct_operators = assemble_position_operators(direct_x, direct_y, basis)
    mode_converter = diagm([cis(k * π / 2) for k ∈ 0:order])
    astig_operators = assemble_position_operators(converted_x, converted_y, basis)
    unitary_transform!(astig_operators, mode_converter)
    operators = compose_povm(direct_operators, astig_operators)
    mtdh = BayesianInference(operators, 10^6, 10^3)
    hermitian_basis = get_hermitian_basis(order + 1)

    @showprogress for (n, m) ∈ Iterators.product(eachindex(photocounts), 1:50)
        outcomes = history2dict(view(histories, 1:photocounts[n], m))
        xs = prediction(outcomes, mtdh) |> mean
        ρ = linear_combination(xs, hermitian_basis)
        ψ = project2pure(ρ)

        all_fids[n, m, k] = abs2(coefficients[:, m] ⋅ ψ)
    end
end

fids = dropdims(mean(all_fids, dims=2), dims=2)
##
out = h5open("Results/Photocount/bayes.h5", "cw")
out["fids"] = fids
out["photocounts"] = photocounts
close(out)