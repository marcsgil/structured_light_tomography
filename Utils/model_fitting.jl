using StructuredLight

function twoD_Gaussian(xy, p)
    amplitude, x₀, y₀, w, α, offset = p

    x = view(xy, 1, :)
    y = view(xy, 2, :)
    @. offset + amplitude * abs2(hg(x - x₀, α * (y - y₀); w, include_normalization=false))
end