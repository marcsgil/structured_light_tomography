using PartiallyCoherentSources, StructuredLight, CairoMakie

rs = LinRange(-3, 3, 512)
basis = stack([hg(rs, rs, m=1), hg(rs, rs, n=1)])
weights = [1 / 2, 1 / 2]

##
N = 300
fields = generate_fields(N, weights, basis, PhaseRandomized())
I = sum(x -> abs2.(x), eachslice(fields, dims=3)) / N
visualize(I)
##

I2 = sum(pair -> pair[1] * abs2.(pair[2]), zip(weights, eachslice(basis, dims=3)))

maximum(I)
maximum(I2)

isapprox(I,I2, rtol = 1e-2)

visualize(sum(pair -> pair[1] * abs2.(pair[2]), zip(weights, eachslice(basis, dims=3))))