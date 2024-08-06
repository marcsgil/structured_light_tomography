using StructuredLight, LinearAlgebra, CairoMakie, ProgressMeter, BayesianTomography
includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")
includet("../Utils/obstructed_measurements.jl")

function fisher_information!(dest, probs, C)
    for n ∈ axes(dest, 2), m ∈ axes(dest, 1)
        tmp = zero(eltype(dest))
        @simd for k ∈ eachindex(probs)
            tmp += C[k, m] * C[k, n] / probs[k]
        end
        dest[m, n] = tmp
    end
end
##
#MUB polarization tomography

bs_povm = [[1.0 0im; 0 0], [0 0; 0 1]] #POVM for a polarazing beam splitter
half_wave_plate = [1 1; 1 -1] / √2 #Unitary matrix for a half-wave plate
quarter_wave_plate = [1 im; 1 -im] / √2 #Unitary matrix for a quarter-wave plate

povm = augment_povm(bs_povm, half_wave_plate, quarter_wave_plate,
    weights=[1 / 3, 1 / 3, 1 / 3])

d = 2
ρs = sample(ProductMeasure(d), 10^5)

C = LinearInversion(povm).C

buffer = Array{Float32,2}(undef, d^2 - 1, d^2 - 1)
traces_inverse_fisher_matrix = Vector{Float64}(undef, size(ρs, 3))

for (m, ρ) ∈ enumerate(eachslice(ρs, dims=3))
    probs = map(povm) do Π
        real(ρ ⋅ Π)
    end
    fisher_information!(buffer, probs, C)
    traces_inverse_fisher_matrix[m] = tr(inv(buffer))
end

mean(traces_inverse_fisher_matrix)
##
#Blade
rs = Base.oneto(200)
x₀ = length(rs) ÷ 2
y₀ = length(rs) ÷ 2
w = length(rs) ÷ 8

basis = positive_l_basis(2, [x₀, y₀, w, 1])
d = length(basis)
ρs = sample(ProductMeasure(d), 10^3)

cutoffs = LinRange(x₀ - w, x₀ + 2.5w, 64)
traces_inverse_fisher_matrix = Matrix{Float64}(undef, size(ρs, 3), length(cutoffs))
##
Threads.@threads for n ∈ eachindex(cutoffs)
    cutoff = cutoffs[n]
    obstructed_basis = get_obstructed_basis(basis, blade_obstruction, cutoff)
    povm, new_ρs = get_proper_povm_and_states(rs, rs, ρs, obstructed_basis)

    filtered_povm = filter(Π -> !iszero(Π), povm)

    C = LinearInversion(filtered_povm).C

    buffer = Array{Float32,2}(undef, d^2 - 1, d^2 - 1)
    for (m, ρ) ∈ enumerate(eachslice(new_ρs, dims=3))
        probs = map(filtered_povm) do Π
            real(ρ ⋅ Π)
        end
        fisher_information!(buffer, probs, C)
        traces_inverse_fisher_matrix[m, n] = tr(inv(buffer))
    end
end

δs = dropdims(mean(traces_inverse_fisher_matrix, dims=1), dims=1)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1],
        xlabel="Blade position (waist)",
        ylabel="Bound on scaled MSE",
        #yscale=log2,
        #yticks=[2^n for n ∈ 0:9],
        xticks=-2:0.5:2.5,
    )
    #ylims!(ax, 1, 100)
    lines!(ax, (cutoffs .- x₀) / w, δs,
        linewidth=3,
    )
    fig
    #save("Plots/fisher_blade.pdf", fig)
end
##
#Iris
rs = Base.oneto(200)
x₀ = length(rs) ÷ 2
y₀ = length(rs) ÷ 2
w = length(rs) ÷ 8

basis = positive_l_basis(2, [x₀, y₀, w, 1])
d = length(basis)
ρs = sample(GinibreEnsamble(d), 10^3)

cutoffs = LinRange(0.1w, 2w, 64)
traces_inverse_fisher_matrix = Matrix{Float32}(undef, size(ρs, 3), length(cutoffs))
##
for n ∈ eachindex(cutoffs)
    cutoff = cutoffs[n]
    obstructed_basis = get_obstructed_basis(basis, iris_obstruction, x₀, y₀, cutoff)
    povm, new_ρs = get_proper_povm_and_states(rs, rs, ρs, obstructed_basis)

    filtered_povm = filter(Π -> !iszero(Π), povm)

    C = LinearInversion(filtered_povm).C

    buffer = Array{Float32,2}(undef, d^2 - 1, d^2 - 1)
    for (m, ρ) ∈ enumerate(eachslice(new_ρs, dims=3))
        """if isone(m)
            display(visualize([real(tr(ρ ⋅ Π)) for Π ∈ povm]))
        end"""

        probs = map(filtered_povm) do Π
            real(ρ ⋅ Π)
        end
        fisher_information!(buffer, probs, C)
        traces_inverse_fisher_matrix[m, n] = tr(inv(buffer))
    end
end

δs = dropdims(mean(traces_inverse_fisher_matrix, dims=1), dims=1)

cutoffs[argmin(δs)] / w
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1],
        xlabel="Iris radius (waist)",
        ylabel="Bound on scaled MSE",
        #yscale=log2,
        #yticks=[2^n for n ∈ 0:9],
        #xticks=-2:0.5:2.5,
    )
    #ylims!(ax, 1, 100)
    lines!(ax, cutoffs / w, δs,
        linewidth=3,
    )
    fig
    save("Plots/fisher_iris.pdf", fig)
end
##
rs = Base.oneto(800)
x₀ = length(rs) ÷ 2
y₀ = length(rs) ÷ 2
w = length(rs) ÷ 8

basis = positive_l_basis(2, [x₀, y₀, w, 1])
d = length(basis)
ρs = reshape([1 0; 0 0], (2, 2, 1))

cutoff = 0.1w
traces_inverse_fisher_matrix = Matrix{Float64}(undef, size(ρs, 3), length(cutoffs))

obstructed_basis = get_obstructed_basis(basis, iris_obstruction, x₀, y₀, cutoff)
povm, new_ρs = get_proper_povm_and_states(rs, rs, ρs, obstructed_basis)

filtered_povm = filter(Π -> !iszero(Π), povm)

C = LinearInversion(filtered_povm).C

buffer = Array{Float32,2}(undef, d^2 - 1, d^2 - 1)

ρ = ρs[:,:,1]

probs = map(filtered_povm) do Π
    real(ρ ⋅ Π)
end


fisher_information!(buffer, probs, C)
tr(inv(buffer))