using StructuredLight

function positive_l_basis(dim, pars)
    x₀ = pars[1]
    y₀ = pars[2]
    w = pars[3]
    [(x, y) -> lg(x .- x₀, y .- y₀; w, p=dim - 1 - n, l=2n) for n ∈ 0:dim-1]
end

function fixed_order_basis(order, pars)
    x₀ = pars[1]
    y₀ = pars[2]
    w = pars[3]
    [(x, y) -> hg(x .- x₀, y - y₀; w, m=order - n, n) for n ∈ 0:order]
end

function only_l_basis(dim, pars)
    x₀ = pars[1]
    y₀ = pars[2]
    w = pars[3]
    [(x, y) -> lg(x .- x₀, y .- y₀; w, l) for l ∈ 0:dim-1]
end