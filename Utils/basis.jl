using StructuredLight

function positive_l_basis!(dest, r, pars)
    dim = length(dest)
    x, y = r
    sqrt_δA, x₀, y₀, w = pars
    for j ∈ eachindex(dest)
        p = dim - j
        l = 2 - 2j # We calculate the negative value of l to get the conjugate
        dest[j] = lg(x - x₀, y - y₀; w, p, l) * sqrt_δA
    end
    dest
end

function positive_l_basis(dim, r, pars)
    buffer = Vector{complex(float(eltype(r)))}(undef, dim)
    positive_l_basis!(buffer, r, pars)
end

function fixed_order_basis!(dest, r, pars, phase=0)
    order = length(dest) - 1
    x, y = r
    sqrt_δA, x₀, y₀, w = pars
    for j ∈ eachindex(dest)
        m = order + 1 - j
        n = j - 1
        dest[j] = hg(x - x₀, y - y₀; w, m, n) * cis(-n * phase) * sqrt_δA
    end
    dest
end

function fixed_order_basis(order, r, pars, phase=0)
    buffer = Vector{complex(float(eltype(r)))}(undef, order + 1)
    fixed_order_basis!(buffer, r, pars, phase)
end