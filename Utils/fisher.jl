function fisher!(F, T, p)
    I = findall(x -> x > 0, vec(p))
    @tullio F[i, j] = T[I[k], i] * T[I[k], j] / p[I[k]]
end

function fisher_at(θ, img, buffer, F, basis_func, Ω, L, ω, rs, T)
    sum(abs2, view(θ, 2:4)) > 1 / 2 && return convert(eltype(θ), NaN)
    ρ = linear_combination(θ, ω)
    get_intensity!(img, buffer, ρ, basis_func, rs, rs)
    normalize!(img, 1)
    fisher!(F, (@view T[:, 2:4]), img)
    J = η_func_jac(θ, Ω, L, ω)[2:end, 2:end]
    tr(inv(J' * F * J))
end