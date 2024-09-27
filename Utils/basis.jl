using StructuredLight

function positive_l_basis(dim, pars)
    x₀ = pars[1]
    y₀ = pars[2]
    w = pars[3]
    [(x, y, N=one(x₀)) -> N * lg(x .- x₀, y .- y₀; w, p=dim - 1 - n, l=2n) for n ∈ 0:dim-1]
end

function fixed_order_basis(order, pars, phase=0.0f0)
    x₀ = pars[1]
    y₀ = pars[2]
    w = pars[3]
    [(x, y, N=one(x₀)) -> N * cis(n * phase) * hg(x .- x₀, y .- y₀; w, m=order - n, n) for n ∈ 0:order]
end