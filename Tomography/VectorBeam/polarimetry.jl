using BayesianTomography, LinearAlgebra, PrettyTables, HDF5

symbols = [:H, :V, :D, :A, :R, :L]
measurements = [get_projector(polarization_state(Val(s))) for s in symbols]

modes = ["Hh", "LG+", "LG-", "PHI+"]
file = h5open("Data/VectorBeam/data.h5")
problem = StateTomographyProblem(measurements)
method = LinearInversion(problem)

ϕs = ([1, 0], [1, -im] / √2, [1, im] / √2, Matrix{Float32}(I, 2, 2) / 2)

for (mode, ϕ) ∈ zip(modes, ϕs)
    p = [(sum(file[joinpath(mode, "I$(pol)_direct")] |> read)
          +
          sum(file[joinpath(mode, "I$(pol)_converted")] |> read)) for pol in symbols]

    normalize!(view(p, 1:2), 1)
    normalize!(view(p, 3:4), 1)
    normalize!(view(p, 5:6), 1)

    if ϕ isa Vector
        c = prediction(p, method)[1] |> project2pure
    else
        c = prediction(p, method)[1]
    end

    printstyled("Mode: $mode \n", color=:blue)
    println("Fidelity: ", round(fidelity(c, ϕ) * 100, digits=3), "%")
end
