
##
using HDF5, ProgressMeter
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
order = 4
histories = file["histories_order$order"] |> read
coefficients = read(file["labels_order$order"])

basis = transverse_basis(order) |> reverse

direct_operators = assemble_position_operators(direct_x, direct_y, basis)
mode_converter = diagm([cis(k * π / 2) for k ∈ 0:order])
astig_operators = assemble_position_operators(converted_x, converted_y, basis)
unitary_transform!(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators)
##
m = 1
photocount = 2048
outcomes = history2dict(view(histories, 1:photocount, m))
xs = sample_posterior(outcomes, operators, 10^6, 10^3)

ρ = linear_combination(xs, get_hermitian_basis(order + 1))
F = eigen(ρ)
ψ = F.vectors[:, order+1]
abs2(coefficients[:, m] ⋅ ψ)