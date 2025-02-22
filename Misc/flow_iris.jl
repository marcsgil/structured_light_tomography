using StructuredLight, CairoMakie, QuantumMeasurements, LinearAlgebra
includet("../Utils/basis.jl")

function obstructed_g(povm, xs, ys, obstruction_func, param...)
    v = first(povm)
    result = zero(v * v')
    for (n, y) ∈ zip(axes(povm, 2), ys)
        for (m, x) ∈ zip(axes(povm, 1), xs)
            result += povm[m, n] * povm[m, n]' * obstruction_func(x, y, param...)
        end
    end
    result
end

function post_measurement_state(θ, g)
    ρ = density_matrix_reconstruction(θ)
    A = sqrt(g)
    σ = A * ρ * A'
    σ /= tr(σ)
    traceless_vectorization(σ)
end

function get_img(ρ, povm, xs, ys, obstruction_func, param...)
    [real(dot(povm[m, n], ρ, povm[m, n])) * obstruction_func(xs[m], ys[n], param...) for m ∈ axes(povm, 1), n ∈ axes(povm, 2)]
end
##
rs = LinRange(-2.5, 2.5, 512)
δA = (rs[2] - rs[1])^2
obstruction_func(x, y, r) = x^2 + y^2 ≤ r^2

povm = [positive_l_basis(2, (x, y), (√δA, 0, 0, 1)) for x ∈ rs, y ∈ rs]

ρ1 = [1 0; 0 0]
ρ2 = [0 0; 0 1]

r = Observable(Inf)
g = @lift(obstructed_g(povm, rs, rs, obstruction_func, $r))
points = @lift([√2 * post_measurement_state([x, 0, z] / √2, $g) for x ∈ LinRange(-1, 1, 31), z ∈ LinRange(-1, 1, 31) if (x^2 + z^2) ≤ 1])

img1 = @lift(get_img(ρ1, povm, rs, rs, obstruction_func, $r))
img2 = @lift(get_img(ρ2, povm, rs, rs, obstruction_func, $r))

radii = reverse(LinRange(0.1, 1.5, 240))

with_theme(theme_latexfonts()) do
    fig = Figure(; size=(1500, 600), fontsize=32)
    ax1 = Axis(fig[1, 1], aspect=DataAspect(), xlabel="x", ylabel="z")
    ax2 = Axis(fig[1, 2], aspect=DataAspect(), title=@lift("r=$(round($r, digits=2)) w"))
    ax3 = Axis(fig[1, 3], aspect=DataAspect())
    hidedecorations!(ax2)
    hidedecorations!(ax3)

    ϕs = LinRange(0, 2π, 512)
    lines!(ax1, cos.(ϕs), sin.(ϕs), color=:black, linestyle=:dash, linewidth=4)
    scatter!(ax1, @lift(first.($points)), @lift(last.($points)), markersize=18)
    heatmap!(ax2, img1, colormap=:jet)
    heatmap!(ax3, img2, colormap=:jet)

    record(fig, "/Users/marcsgil/Code/Presentation-Tomography-SL/Images/flow_iris.mp4", radii; framerate=24) do radius
        r[] = radius
    end
end