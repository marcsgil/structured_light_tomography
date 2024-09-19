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
    x₀, y₀, w, amplitude, offset = p
    offset + amplitude * abs2(hg(x - x₀, y - y₀; w))
end

function calibration_fit(x, y, calibration::AbstractMatrix)
    p0 = [0, 0, 0.15f0, maximum(calibration), minimum(calibration)]
    surface_fit(gaussian_model, x, y, calibration, p0)
end

function calibration_fit(x, y, calibration)
    (calibration_fit(x, y, slice) for slice ∈ eachslice(calibration, dims=3))
end