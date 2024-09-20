using HDF5, BayesianTomography, ProgressMeter, CairoMakie

includet("../../Utils/model_fitting.jl")
includet("../../Utils/basis.jl")
includet("../../Utils/position_operators.jl")

function load_data(path, mode_name, symbols, states)
    imgs = h5open(path) do file
        stack(
            read(file[mode_name*"/I$(pol)_$state"]) for pol ∈ symbols, state ∈ states
        )
    end

    for j ∈ 1:2:size(imgs, 3)
        normalize!(view(imgs, :, :, j:j+1, :), 1)
    end

    remove_background!(imgs)

    imgs
end

function get_measurements(imgs, pol_measurements, order)
    pars = stack(
        center_of_mass_and_waist(slice, 1) for slice ∈ eachslice(imgs, dims=(3, 4))
    )

    position_measurements = Array{Matrix{ComplexF32}}(undef, size(imgs)...)

    for n ∈ axes(position_measurements, 3)
        direct_basis = fixed_order_basis(1, view(pars, :, n, 1))
        converted_basis = [(x, y) -> f(x, y) * cis(-(k - 1) * π / 2)
                           for (k, f) ∈ enumerate(fixed_order_basis(1, view(pars, :, n, 2)))]

        if order[n]
            J = (1, 2)
        else
            J = (2, 1)
        end

        position_measurements[:, :, n, J[1]] = assemble_position_operators(xs, ys, direct_basis)
        position_measurements[:, :, n, J[2]] = assemble_position_operators(xs, ys, converted_basis)
    end

    [
        kron(position_measurements[J], pol_measurements[J[3]])
        for J ∈ eachindex(IndexCartesian(), position_measurements)
    ]
end
##
symbols = [:H, :V, :D, :A, :R, :L]
pol_measurements = [get_projector(polarization_state(Val(s))) for s in symbols]

states = ["left", "right"]
##
modes = Dict(
    "Hh" => kron([1, 0], [1, 0]),
    "LG+" => kron([1, im], [1, -im]) / 2,
    "LG-" => kron([1, -im], [1, +im]) / 2,
    "PHI+" => [1, 0, 0, 1] / √2
)

order = Dict(
    "Hh" => (true, true, true, true, true, true),
    "LG+" => (true, true, true, true, true, true),
    "LG-" => (false, true, false, true, true, true),
    "PHI+" => (true, true, false, true, false, true)
)

path = "Data/VectorBeam/cropped_data.h5"
##
mode_name = "Hh"
ϕ = modes[mode_name]

imgs = load_data(path, mode_name, symbols, states);
measurement = get_measurements(imgs, pol_measurements, order[mode_name]);

sim = simulate_outcomes(ϕ, measurement, 10^6)

display(visualize(imgs))
visualize(sim)
##
for (mode, ϕ) ∈ modes
    imgs = load_data(path, mode, symbols, states)

    measurement = get_measurements(imgs, pol_measurements, order[mode])

    problem = StateTomographyProblem(measurement)
    method = MaximumLikelihood(problem)
    ϕ_pred = prediction(imgs, method)[1] |> project2pure
    printstyled("Mode: $mode \n", color=:blue)
    println("Fidelity: ", round(fidelity(ϕ, ϕ_pred) * 100, digits=3), "%")
end
##
modes = Dict(
    "LG(-)" => kron([1, -im], [1, im]) / 2,
)

order = Dict(
    "LG(-)" => (true, true, true, true, true, true),
)

path = "Data/VectorBeam3/cropped_data.h5"
##
mode_name = "LG(-)"
ϕ = modes[mode_name]

imgs = load_data(path, mode_name, symbols, states);
measurement = get_measurements(imgs, pol_measurements, order[mode_name]);

sim = simulate_outcomes(ϕ, measurement, 10^6)

display(visualize(imgs))
visualize(sim)
##
for (mode, ϕ) ∈ modes
    imgs = load_data(path, mode, symbols, states)

    measurement = get_measurements(imgs, pol_measurements, order[mode])

    problem = StateTomographyProblem(measurement)
    method = MaximumLikelihood(problem)
    ϕ_pred = prediction(imgs, method)[1] |> project2pure
    printstyled("Mode: $mode \n", color=:blue)
    println("Fidelity: ", round(fidelity(ϕ, ϕ_pred) * 100, digits=3), "%")
end