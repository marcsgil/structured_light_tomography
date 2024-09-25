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
    offset + amplitude * exp(-2 * ((x - x₀)^2 + (y - y₀)^2) / w^2)
end

function calibration_fit(x, y, calibration::AbstractMatrix)
    Δx = x[end] - x[begin]
    Δy = y[end] - y[begin]
    w = max(Δx / 4, Δy / 4)
    x₀ = x[length(x)÷2]
    y₀ = y[length(y)÷2]
    p0 = [x₀, y₀, w, maximum(calibration), minimum(calibration)]
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

function center_of_mass_and_variance(img)
    T = float(typeof(firstindex(img)))

    m₀ = zero(T)
    n₀ = zero(T)
    s² = zero(T)
    N = zero(T)

    for n ∈ axes(img, 2), m ∈ axes(img, 1)
        m₀ += m * img[m, n]
        n₀ += n * img[m, n]
        s² += (m^2 + n^2) * img[m, n]
        N += img[m, n]
    end

    m₀ /= sum(img)
    n₀ /= sum(img)
    s² = s² / N - m₀^2 - n₀^2

    m₀, n₀, s²
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