using StructuredLight

function positive_l_basis(dim, pars)
    x₀ = pars[1]
    y₀ = pars[2]
    γ = pars[3]
    α = pars[4]
    [(x, y) -> lg(x .- x₀, α .* (y .- y₀); γ, p=dim - 1 - n, l=2n) for n ∈ 0:dim-1]
end

function fixed_order_basis(order, pars)
    x₀ = pars[1]
    y₀ = pars[2]
    γ = pars[3]
    α = pars[4]
    [(x, y) -> hg(x - x₀, α * (y - y₀); γ, m, n=order - m) for m ∈ 0:order]
end

function only_l_basis(dim, pars)
    x₀ = pars[1]
    y₀ = pars[2]
    γ = pars[3]
    α = pars[4]
    [(x, y) -> lg(x .- x₀, α .* (y .- y₀); γ, l) for l ∈ 0:dim-1]
end