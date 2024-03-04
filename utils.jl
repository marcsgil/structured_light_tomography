using ClassicalOrthogonalPolynomials

function hg(x, y, m, n)
    N = 1 / sqrt(2^(m + n) * factorial(m) * factorial(n) * π)
    N * hermiteh(m, x) * hermiteh(n, y) * exp(-(x^2 + y^2) / 2)
end

function fidelity(ρ, σ)
    sqrt_ρ = sqrt(ρ)
    abs2(tr(sqrt(sqrt_ρ * σ * sqrt_ρ)))
end