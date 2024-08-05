using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter
includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("../Utils/obstructed_measurements.jl")

import BayesianTomography: ProductMeasure, LinearInversion

function fisher_information!(dest, probs::AbstractVector, C::AbstractMatrix)
    for I ∈ eachindex(IndexCartesian(), dest)
        tmp = zero(eltype(dest))
        for k ∈ eachindex(probs)
            tmp += C[k, I[1]] * C[k, I[2]] / probs[k]
        end
        dest[I] = tmp
    end
end
##
rs = Base.oneto(200)
x₀ = length(rs) ÷ 2
y₀ = length(rs) ÷ 2
w = length(rs) ÷ 8

basis = positive_l_basis(2, [x₀, y₀, w, 1])
##


##
cutoffs = LinRange(x₀ - w, x₀ + 2.5w, 16)

N = 10^3
d = length(basis)
ρs = sample(ProductMeasure(d), N)

Is = Array{Float32,4}(undef, d^2 - 1, d^2 - 1, N, length(cutoffs))

@showprogress for n ∈ eachindex(cutoffs)
    obstructed_basis = get_obstructed_basis(basis, blade_obstruction, cutoffs[n])

    povm, ρs = get_proper_povm_and_states(rs, rs, ρs, obstructed_basis)
    filtered_povm = filter(Π -> !iszero(Π), povm)

    C = LinearInversion(povm).C

    for (m, ρ) ∈ enumerate(eachslice(ρs, dims=3))
        probs = vec([real(ρ ⋅ Π) for Π in filtered_povm])
        fisher_information!(view(Is, :, :, m, n), probs, C)
    end
end

mean_Is = dropdims(mean(Is, dims=3), dims=3)

inv_Is = mapslices(inv, mean_Is, dims=(1, 2))
diag_inv_Is = dropdims(mapslices(diag, inv_Is, dims=(1, 2)), dims=2)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1],
        xlabel="Blade position (waist)",
        ylabel="Average inverse Fisher information",
        #yscale=log2,
        #yticks=[2^n for n ∈ 0:13],
        xticks=-2:0.5:2.5,
    )
    #yscale=log10)
    #ylims!(ax, 1, 100)
    series!(ax, (cutoffs .- x₀) / w, diag_inv_Is,
        labels=[L"(I^{-1})_{XX}", L"(I^{-1})_{YY}", L"(I^{-1})_{ZZ}"],
        color=[:red, :green, :blue],
        linewidth=3,
        linestyle=[:solid, :dash, :dot],)
    axislegend(position=:rt)
    fig
    #save("Plots/fisher_blade.pdf", fig)
end
##
cutoffs = LinRange(0.1w, 2w, 32)

N = 10^3
d = length(_basis)
ρs = sample(ProductMeasure(d), N)

Is = Array{Float32,3}(undef, d^2 - 1, d^2 - 1, length(cutoffs))

for (n, cutoff) ∈ enumerate(cutoffs)
    obstructed_basis = get_obstructed_basis(basis, iris_obstruction, x₀, y₀, cutoffs[n])
    povm, ρs = get_proper_povm_and_states(rs, rs, ρs, obstructed_basis)
    Is[:, :, n] = mean(fisher_information(ρs, povm), dims=3)
end

inv_Is = mapslices(inv, Is, dims=(1, 2))
diag_inv_Is = dropdims(mapslices(diag, inv_Is, dims=(1, 2)), dims=2)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1],
        xlabel="Iris radius (waist)",
        ylabel="Average inverse Fisher information",
        #yscale=log2,
        #yticks=[2^n for n ∈ 0:13],
    )
    #yscale=log10)
    ylims!(ax, 0, 3)
    series!(ax, cutoffs / w, diag_inv_Is,
        labels=[L"(I^{-1})_{XX}", L"(I^{-1})_{YY}", L"(I^{-1})_{ZZ}"],
        color=[:red, :green, :blue],
        linestyle=[:solid, :dash, :dot],
        linewidth=3,)
    axislegend(position=:rt)
    fig
    #save("Plots/fisher_iris.pdf", fig)
end
##
rs = Base.oneto(200)
x₀ = length(rs) ÷ 2
y₀ = length(rs) ÷ 2
w = length(rs) ÷ 4

basis = only_l_basis(4, [x₀, y₀, w, 1])

povm = assemble_position_operators(rs, rs, basis)

N = 10^3
d = length(basis)
ρs = sample(ProductMeasure(d), N)

inv_I = mean(ρ -> fisher_information(ρ, povm), eachslice(ρs, dims=3)) |> inv

diag(inv_I) |> sort


ρs[:, :, 5]
visualize([real(tr(ρs[:, :, 6] * Π)) for Π ∈ povm])
##
