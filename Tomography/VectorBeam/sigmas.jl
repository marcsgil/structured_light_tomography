using LinearAlgebra, PrettyTables, Images, HDF5, BayesianTomography

pol_name = ["h", "v", "d", "a", "r", "L"]
modes = ["Hh", "LG+", "LG-", "PHI+"]
expecteds = [[1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]]
formatter = (v, i, j) -> j > 1 ? round(v, digits=3) : v
file = h5open("ExperimentalData/data.h5")

bloch2density(Σ) = [1+Σ[3] Σ[1]-im*Σ[2]; Σ[1]+im*Σ[2] 1-Σ[3]] / 2

for (mode, expected) ∈ zip(modes, expecteds)
    p = Vector{Float64}(undef, 6)
    for (i, pol) in enumerate(["h", "v", "d", "a", "r", "L"])
        for side ∈ ["right", "left"]
            p[i] = sum(read(file["$mode/I$(pol)_$(side)"]))
        end
    end

    paths = [joinpath("ExperimentalData", mode, "I$(pol).jpg") for pol ∈ pol_name]
    imgs = [load(path) for path in paths]
    p = [sum(x -> x.val, img) for img ∈ imgs]

    normalize!(view(p, 1:2), 1)
    normalize!(view(p, 3:4), 1)
    normalize!(view(p, 5:6), 1)
    Σ = [p[1] - p[2], p[3] - p[4], p[5] - p[6]]

    data = hcat(expected, Σ)

    if isposdef(bloch2density(Σ))
        fid = fidelity(bloch2density(Σ), bloch2density(expected)) * 100
        printstyled("Mode: $mode; Fidelity: $(round(fid, sigdigits=3))% \n", color=:blue)
    else
        printstyled("Mode: $mode; Not positive semi definite \n", color=:blue)
    end


    pretty_table(hcat(["z", "x", "y"], data), header=["σ", "Theo.", "Exp."], formatters=formatter)
end
##
paths = [joinpath("ExperimentalData", "Hh", "I$(pol).jpg") for pol ∈ pol_name]
imgs = [load(path) for path in paths]

p = [sum(x -> x.val, img) for img ∈ imgs]