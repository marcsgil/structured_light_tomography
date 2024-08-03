using PositionMeasurements, CairoMakie, HDF5, StructuredLight, Tullio, PositionMeasurements
using LinearAlgebra

includet("../Utils/basis.jl")
file = h5open("Data/Raw/positive_l.h5")

rs = Base.oneto(200)
x₀ = length(rs) ÷ 2
y₀ = length(rs) ÷ 2
w = length(rs) ÷ 8
γ = w / √2

dim = 3

basis = positive_l_basis(dim, [x₀, y₀, γ, 1])
povm = assemble_position_operators(rs, rs, basis)
##
n = 100

exp = file["images_dim$dim"][:, :, n]
ρ = file["labels_dim$dim"][:, :, n]
theo = [real(tr(ρ * Π)) for Π ∈ povm]

visualize(stack([exp,theo]))