using StructuredLight

function positive_l_basis(dim, w)
    [(x, y) -> lg(x, y; w, p=dim - 1 - n, l=2n) for n ∈ 0:dim-1]
end