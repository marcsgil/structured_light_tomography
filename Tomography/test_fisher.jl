using BayesianTomography, StructuredLight,
    PositionMeasurements, LinearAlgebra, CairoMakie,
    Tullio

function mean_fisher_diag!(Is, ρs,
    povm::AbstractArray{T,3}, rs, Rs) where {T}
    Δr = rs[2] - rs[1]

    basis = gell_mann_matrices(size(ρs, 1), include_identity=false)
    C = [real_orthogonal_projection(E, basis) for E ∈ povm] |> stack

    Threads.@threads for n ∈ axes(ρs, 3)
        P = [real(view(ρs, :, :, n) ⋅ E) for E in povm]

        for (m, R) ∈ enumerate(Rs)
            idx = Integer(fld(R - minimum(rs), Δr)) + 1
            _C = @view C[:, idx:end, :, :]
            _P = normalize((@view P[idx:end, :, :]), 1)

            for i ∈ axes(Is, 1)
                for l ∈ axes(_P, 3), k ∈ axes(_P, 2), j ∈ axes(_P, 1)
                    Is[i, m, n] += _C[i, j, k, l] * _C[i, j, k, l] / _P[j, k, l]
                end
            end
        end
    end
end

function mean_fisher_diag!(Is, ρs,
    povm::AbstractMatrix{T}, rs, Rs) where {T}
    Δr = rs[2] - rs[1]

    basis = gell_mann_matrices(size(ρs, 1), include_identity=false)
    C = [real_orthogonal_projection(E, basis) for E ∈ povm] |> stack

    Threads.@threads for n ∈ axes(ρs, 3)
        P = [real(view(ρs, :, :, n) ⋅ E) for E in povm]

        for (m, R) ∈ enumerate(Rs)
            idx = Integer(fld(R - minimum(rs), Δr)) + 1
            _C = @view C[:, idx:end, :]
            _P = normalize((@view P[idx:end, :]), 1)

            for i ∈ axes(Is, 1)
                for k ∈ axes(_P, 2), j ∈ axes(_P, 1)
                    Is[i, m, n] += _C[i, j, k] * _C[i, j, k] / _P[j, k]
                end
            end
        end
    end
end

function mean_fisher_diag!(Is, ρs,
    povm::AbstractVector{T}, rs, Rs) where {T}
    Δr = rs[2] - rs[1]

    basis = gell_mann_matrices(size(ρs, 1), include_identity=false)
    C = [real_orthogonal_projection(E, basis) for E ∈ povm] |> stack

    Threads.@threads for n ∈ axes(ρs, 3)
        P = [real(view(ρs, :, :, n) ⋅ E) for E in povm]

        for (m, R) ∈ enumerate(Rs)
            idx = Integer(fld(R - minimum(rs), Δr)) + 1
            _C = @view C[:, idx:end]
            _P = normalize((@view P[idx:end]), 1)

            for i ∈ axes(Is, 1)
                for j ∈ axes(_P, 1)
                    Is[i, m, n] += _C[i, j] * _C[i, j] / _P[j]
                end
            end
        end
    end
end
##
rs = LinRange(-4, 4, 256)
Rs = LinRange(-3, 2, 32)
Δr = rs[2] - rs[1]

"""basis = [(r, par) -> hg(r[1], r[2]; m=1),
    (r, par) -> hg(r[1], r[2]; n=1)]
direct_povm = assemble_position_operators(rs, rs, basis)
converted_povm = assemble_position_operators(rs, rs, basis)
mode_converter = diagm([cis(-k * π / 2) for k ∈ 0:1])
unitary_transform!(converted_povm, mode_converter)
povm = compose_povm(direct_povm, converted_povm)"""

"""basis = [(r, par) -> lg(r[1], r[2]; p=1, l=0),
    (r, par) -> lg(r[1], r[2]; l=2)]
povm = assemble_position_operators(rs, rs, basis)"""

##
outer(v) = v * v'

povm = outer.([
    [1, 0],
    [0, 1],
    [1, 1] / √2,
    [1, -1] / √2,
    [1, -im] / √2,
    [1, im] / √2
]) / 3


mean([fisher_information(ρ, povm) for ρ ∈ eachslice(sample(GinibreEnsamble(2), 10^6), dims=3)])
##
##
N = 10^3

ρs = sample(GinibreEnsamble(2), N)
Is = zeros(Float32, 3, length(Rs), size(ρs, 3))

mean_fisher_diag!(Is, ρs, direct_povm, rs, Rs)
##
@code_warntype mean_fisher_diag!(Is, ρs, povm, rs, Rs)
@btime mean_fisher_diag!($Is, $ρs, $povm, $rs, $Rs)
##
with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)
    ax = Axis(fig[1, 1],
        xlabel="Blade position (waist)",
        ylabel="Average Fisher information",
        xticks=first(Rs):last(Rs),
        yticks=0:0.1:2,
        title="MUB",
    )
    #yscale=log10)
    #ylims!(ax, 0, 1.3)
    series!(ax, Rs, dropdims(mean(Is, dims=3), dims=3),
        labels=[L"I_{XX}", L"I_{YY}", L"I_{ZZ}"],
        color=[:red, :green, :blue],
        linewidth=3,)
    axislegend()
    fig
    save("Plots/fisher_mub.png", fig)
end

#dot