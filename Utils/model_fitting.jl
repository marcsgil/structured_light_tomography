using StructuredLight, LsqFit

function surface_fit(model, x, y, data, p0)
    function _model(xy, p)
        x = view(xy, 1, :)
        y = view(xy, 2, :)
        map((x, y) -> model(x, y, p), x, y)
    end

    xy = hcat(([x, y] for x in x, y in y)...)
    LsqFit.curve_fit(_model, xy, vec(data), p0)
end

function gaussian_model(x, y, p)
    x₀, y₀, w, α, amplitude, offset = p
    offset + amplitude * abs2(hg(x - x₀, α * (y - y₀); w))
end

function blade_model(x, y, p)
    _p = @view p[begin:end-1]
    blade_pos = p[end]
    gaussian_model(x, y, _p) * (x < blade_pos)
end