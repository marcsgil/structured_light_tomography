using LuxUtils

function get_model()
    Chain(
        Conv((5, 5), 1 => 24, elu),
        MeanPool((2, 2)),
        Conv((5, 5), 24 => 40, elu),
        MeanPool((2, 2)),
        Conv((5, 5), 40 => 35, elu),
        MeanPool((2, 2)),
        FlattenLayer(),
        Dense(560, 120, elu),
        Dense(120, 80, elu),
        Dense(80, 40, elu),
        Dense(40, 3, identity)
    )
end

function normalize_data!(x, dims)
    x .-= mean(x, dims=dims)
    x ./= std(x, dims=dims)
    nothing
end