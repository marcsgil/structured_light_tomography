using StructuredLight

function positive_l_basis(dim, x₀, y₀, w, α)
    [(x, y) -> lg(x - x₀, α * (y - y₀); w, p=dim - 1 - n, l=2n, include_normalization=false) for n ∈ 0:dim-1]
end

function fixed_order_basis(order, x₀, y₀, w, α)
    [(x, y) -> hg(x - x₀, α * (y - y₀); w, m, n=order - m, include_normalization=false) for m ∈ 0:order]
end