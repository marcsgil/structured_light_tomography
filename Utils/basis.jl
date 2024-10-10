using StructuredLight

"""function positive_l_basis(dim, pars)
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
end"""

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
        dest[j] = hg(x - x₀, y - y₀; w, m, n) * cis(-j * phase) * sqrt_δA
    end
    dest
end

function fixed_order_basis(order, r, pars, phase=0)
    buffer = Vector{complex(float(eltype(r)))}(undef, order + 1)
    positive_l_basis!(buffer, r, pars, phase)
end