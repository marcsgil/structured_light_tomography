using BayesianTomography, HDF5, ProgressMeter

includet("../Data/data_treatment_utils.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")

file = h5open("Data/Processed/pure_photocount.h5")

direct_lims = read(file["direct_lims"])
converted_lims = read(file["converted_lims"])
direct_x, direct_y = get_grid(direct_lims, (64, 64))
converted_x, converted_y = get_grid(converted_lims, (64, 64))
##
order = 1

histories = file["histories_order$order"] |> read
coefficients = read(file["labels_order$order"])

basis = fixed_order_basis(order, [0, 0, 1 / √2, 1])

direct_operators = assemble_position_operators(direct_x, direct_y, basis)
mode_converter = diagm([cis(Float32(k * π / 2)) for k ∈ 0:order])
astig_operators = assemble_position_operators(converted_x, converted_y, basis)
unitary_transform!(astig_operators, mode_converter)
operators = compose_povm(direct_operators, astig_operators);
problem = StateTomographyProblem(operators)
mthd = BayesianInference(problem)
##
orders = 1:4
photocounts = [2^k for k ∈ 6:11]
all_fids = zeros(Float64, length(photocounts), 50, length(orders))

p = Progress(length(all_fids))
for (k, order) ∈ enumerate(orders)
    histories = file["histories_order$order"] |> read
    coefficients = read(file["labels_order$order"])

    basis = fixed_order_basis(order, [0, 0, √2, 1])

    direct_operators = assemble_position_operators(direct_x, direct_y, basis)
    mode_converter = diagm([cis(Float32(k * π / 2)) for k ∈ 0:order])
    astig_operators = assemble_position_operators(converted_x, converted_y, basis)
    unitary_transform!(astig_operators, mode_converter)
    operators = compose_povm(direct_operators, astig_operators)
    problem = StateTomographyProblem(operators)
    mthd = BayesianInference(problem)

    for m ∈ 1:50
        for n ∈ eachindex(photocounts)
            outcomes = complete_representation(History(view(histories, 1:photocounts[n], m)), (64, 64, 2))
            ρ, _ = prediction(outcomes, mthd)
            ψ = project2pure(ρ)

            all_fids[n, m, k] = abs2(coefficients[:, m] ⋅ ψ)
            next!(p)
        end
    end
end

fids = dropdims(mean(all_fids, dims=2), dims=2)
##
out = h5open("Results/Photocount/bayes.h5", "cw")
out["fids"] = fids
out["photocounts"] = photocounts
close(out)