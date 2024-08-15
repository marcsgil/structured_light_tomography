using Integrals, StructuredLight, CairoMakie, LinearAlgebra

function X(u)
    √2 * real(lg(u[1], u[2], p=1) * lg(u[1], u[2], l=2))
end

function Y(u)
    √2 * imag(lg(u[1], u[2],p=1) * lg(u[1], u[2],l=2))
end

function Z(u)
    (abs2(lg(u[1], u[2], p=1)) - abs2(lg(u[1], u[2], l=2))) / √2
end

function Id(u)
    abs2(lg(u[1], u[2], p=1)) + abs2(lg(u[1], u[2], l=2))
end

function γ(u, θ)
    Id(u) / 2 + θ[1] * X(u) + θ[2] * Y(u) + θ[3] * Z(u)
end

const basis_func = [X, Y, Z]

function integrand(u, par)
    θ, j, k = par
    basis_func[j](u) * basis_func[k](u) / γ(u, θ)
end
##


sum(Z(r)^2 / γ(r, [0,0,0]) for r ∈ Iterators.product(rs,rs)) * (rs[2] - rs[1])^2


heatmap([Y(r)^2 for r ∈ Iterators.product(rs,rs)], colormap = :coolwarm)
##
fisher = Matrix{Float64}(undef, 3, 3)
domain = (fill(-3, 2), fill(3,2)) # (lb, ub)
for n ∈ axes(fisher,2), m ∈ axes(fisher,1)
    par = ([0.1,0.1,0.3], m,n)
    prob = IntegralProblem(integrand, domain, par)
    sol = solve(prob, HCubatureJL(); reltol = 1e-6, abstol = 1e-6)
    fisher[m,n] = sol.u
end

fisher |> inv |> tr