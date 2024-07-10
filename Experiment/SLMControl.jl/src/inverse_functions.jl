function inverse(f, y, bracket)
    find_zero(x -> f(x) - y, bracket)
end

function interpolated_inverse(f, domain, N=512)
    xs = LinRange(f(domain[1]), f(domain[2]), N)
    ys = [inverse(f, x, domain) for x ∈ xs]

    LinearInterpolation(xs, ys)
end

inverse_besselj1 = interpolated_inverse(besselj1, (0, 1.84))