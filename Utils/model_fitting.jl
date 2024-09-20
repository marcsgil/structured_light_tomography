using StructuredLight, LsqFit, StatsBase

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

function center_of_mass(img::AbstractMatrix{T}) where {T}
    m₀ = zero(T)
    n₀ = zero(T)
    for n ∈ axes(img, 2), m ∈ axes(img, 1)
        m₀ += m * img[m, n]
        n₀ += n * img[m, n]
    end
    m₀ / sum(img), n₀ / sum(img)
end

function center_of_mass_and_variance(img::AbstractMatrix{T}) where {T}
    m₀, n₀ = center_of_mass(img)

    r2 = zero(T)
    for n ∈ axes(img, 2), m ∈ axes(img, 1)
        r2 += ((m - m₀)^2 + (n - n₀)^2) * img[m, n]
    end

    m₀, n₀, r2 / sum(img)
end

function center_of_mass_and_waist(img, order)
    m₀, n₀, s² = center_of_mass_and_variance(img)
    m₀, n₀, √(2 * s² / (order + 1))
end

most_frequent_value(data) = countmap(data) |> argmax

function remove_background!(images, bg)
    map!(x -> x < bg ? zero(x) : x - bg, images, images)
end

function remove_background!(images)
    bg = most_frequent_value(images)
    remove_background!(images, bg)
end