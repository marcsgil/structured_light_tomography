function blade_obstruction(x, y, blade_pos)
    x < blade_pos
end

function iris_obstruction(x, y, x₀, y₀, radius)
    (x - x₀)^2 + (y - y₀)^2 < radius^2
end

function inverse_iris_obstruction(x, y, x₀, y₀, radius)
    !iris_obstruction(x, y, x₀, y₀, radius)
end

function get_obstructed_basis(basis, obstruction_func, args...; kwargs...)
    map(basis) do f
        (x, y) -> f(x, y) * obstruction_func(x, y, args...; kwargs...)
    end
end

function get_valid_indices(x, y, obstruction_func, args...; kwargs...)
    findall(r -> obstruction_func(r..., args...; kwargs...), collect(Iterators.product(x, y)))
end