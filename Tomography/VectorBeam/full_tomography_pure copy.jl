using HDF5, BayesianTomography, ProgressMeter, CairoMakie

includet("../../Utils/model_fitting.jl")
includet("../../Utils/basis.jl")
includet("../../Utils/position_operators.jl")

symbols = [:H, :V, :D, :A, :R, :L]
pol_measurements = [get_projector(polarization_state(Val(s))) for s in symbols]

modes = ["Hh", "LG+", "LG-", "PHI+"]
states = ["direct", "converted"]

ϕs = [kron([1, 0], [1, 0]),
    kron([1, -im], [1, -im]) / 2,
    kron([1, im], [1, +im]) / 2,
    [1, 0, 0, -1] / √2]

h5open("Data/VectorBeam/data.h5") do file
    for (mode, ϕ) ∈ zip(modes, ϕs)
        imgs = stack(
            read(file[mode*"/I$(pol)_$state"]) for pol ∈ symbols, state ∈ states
        )

        for j ∈ 1:2:size(imgs, 3)
            normalize!(view(imgs, :, :, j:j+1, :), 1)
        end


        xs = axes(imgs, 1)
        ys = axes(imgs, 2)

        remove_background!(imgs)

        pars = stack(
            center_of_mass_and_waist(slice, 1) for slice ∈ eachslice(imgs, dims=(3, 4))
        )

        position_measurements = Array{Matrix{ComplexF32}}(undef, size(imgs)...)

        for n ∈ axes(position_measurements, 3)
            direct_basis = fixed_order_basis(1, view(pars, :, n, 1))
            converted_basis = [(x, y) -> f(x, y) * cis((k - 1) * π / 2)
                               for (k, f) ∈ enumerate(fixed_order_basis(1, view(pars, :, n, 2)))]


            position_measurements[:, :, n, 1] = assemble_position_operators(xs, ys, reverse(direct_basis))
            position_measurements[:, :, n, 2] = assemble_position_operators(xs, ys, reverse(converted_basis))
        end

        measurement = [
            kron(position_measurements[J], pol_measurements[J[3]])
            for J ∈ eachindex(IndexCartesian(), position_measurements)
        ]

        problem = StateTomographyProblem(measurement)
        method = MaximumLikelihood(problem)
        ϕ_pred = prediction(imgs, method)[1] |> project2pure
        printstyled("Mode: $mode \n", color=:blue)
        println("Fidelity: ", round(fidelity(ϕ, ϕ_pred) * 100, digits=3), "%")
        #display(ϕ_pred)
    end
end
##
m = 4

imgs = h5open("Data/VectorBeam/data.h5") do file
    stack(
        read(file[modes[m]*"/I$(pol)_$state"]) for pol ∈ symbols, state ∈ states
    )
end

for j ∈ 1:2:size(imgs, 3)
    normalize!(view(imgs, :, :, j:j+1, :), 1)
end

xs = axes(imgs, 1)
ys = axes(imgs, 2)

remove_background!(imgs)

pars = stack(
    center_of_mass_and_waist(slice, 1) for slice ∈ eachslice(imgs, dims=(3, 4))
)

position_measurements = Array{Matrix{ComplexF32}}(undef, size(imgs)...)

for n ∈ axes(position_measurements, 3)
    direct_basis = fixed_order_basis(1, view(pars, :, n, 1))
    converted_basis = [(x, y) -> f(x, y) * cis((k - 1) * π / 2)
                       for (k, f) ∈ enumerate(fixed_order_basis(1, view(pars, :, n, 2)))]


    position_measurements[:, :, n, 1] = assemble_position_operators(xs, ys, reverse(direct_basis))
    position_measurements[:, :, n, 2] = assemble_position_operators(xs, ys, reverse(converted_basis))
end

measurement = [
            kron(position_measurements[J], pol_measurements[J[3]])
            for J ∈ eachindex(IndexCartesian(), position_measurements)
        ]

sim = simulate_outcomes([1, 0, 0, -1] / √2, measurement, 10^6)

display(visualize(imgs))
visualize(sim)
