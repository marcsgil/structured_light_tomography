using Images, BayesianTomography, LinearAlgebra, PrettyTables, LaTeXStrings, HDF5

outer(v) = v * v'

polarization_povm = outer.([
    [1, 0],
    [0, 1],
    [1, 1] / √2,
    [1, -1] / √2,
    [1, -im] / √2,
    [1, im] / √2
]) / 3

pol_name = ["H", "V", "D", "A", "R", "L"]
modes = ["Hh", "LG+", "LG-", "PHI+"]
file = h5open("ExperimentalData/data.h5")
mthd = LinearInversion(polarization_povm)

ϕs = ([1, 0], [1, -im] / √2, [1, im] / √2, Matrix{Float32}(I, 2, 2) / 2)

for (mode, ϕ) ∈ zip(modes, ϕs)
    p = [(sum(file[joinpath(mode, "I$(pol)_direct")] |> read)
          +
          sum(file[joinpath(mode, "I$(pol)_converted")] |> read)) for pol in pol_name]

    normalize!(view(p, 1:2), 1)
    normalize!(view(p, 3:4), 1)
    normalize!(view(p, 5:6), 1)

    if ϕ isa Vector
        c = prediction(p, mthd) |> project2pure
    else
        c = prediction(p, mthd)
    end

    printstyled("Mode: $mode \n", color=:blue)
    println("Fidelity: ", round(fidelity(c, ϕ) * 100, digits=3), "%")
end
