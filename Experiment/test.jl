using StructuredLight, Tullio, LinearAlgebra, BayesianTomography

function get_basis(x, y, order, w0)
    stack((lg(x, y; w0, p, l=order - 2p) for p ∈ 0:order÷2))
end

function get_fields(cs, basis)
    @tullio fields[n, j, k] := basis[j, k, m] * cs[m, n]
end

function get_fields_and_probs(ρ, basis)
    ps, cs = eigen(Hermitian(ρ))
    get_fields(cs, basis), ps
end