using Random, StatsBase

function get_events(x)
    events = Vector{Int}(undef, sum(Int, x))
    counter = 1
    for n ∈ eachindex(x)
        for _ ∈ 1:x[n]
            events[counter] = n
            counter += 1
        end
    end
    events
end

function sample_events(rng, x, N)
    events = get_events(x)
    sampled_events = StatsBase.sample(rng, events, N, replace=false)
    result = zero(x)
    for (k, v) ∈ countmap(sampled_events)
        result[k] = v
    end
    result
end

sample_events(x, N) = sample_events(Xoshiro(0), x, N)