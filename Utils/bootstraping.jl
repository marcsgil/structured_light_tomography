using Statistics

function bootstrap(data::AbstractArray{T};
    n_bootstraps=10^5, ci_level=0.95) where {T}
    n = length(data)
    sample_mean = mean(data)

    bootstrap_means = [mean(rand(data, n)) for _ in 1:n_bootstraps]

    lower = T((1 - ci_level) / 2)
    upper = 1 - lower

    ci = quantile(bootstrap_means, (lower, upper))

    sample_mean, ci...
end