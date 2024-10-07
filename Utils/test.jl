using StructuredLight, BayesianTomography, BenchmarkTools
using Base.Threads: @spawn

function positive_l_basis!(dest, x, y, x₀, y₀, w)
    dim = length(dest)
    for n ∈ eachindex(dest)
        p = dim - n
        l = 2n - 2
        dest[n] = lg(x - x₀, y - y₀; w, p, l)
    end
end

function get_decomposition!(buffer, traceless_part_slices, trace_part, rs, args...)
    for (n, r, slice) ∈ zip(eachindex(trace_part), rs, traceless_part_slices)
        positive_l_basis!(buffer, r..., args...)
        gell_mann_projection!(slice, buffer)
        trace_part[n] = sum(abs2, buffer)
    end
end

function get_decomposition!(buffers::AbstractMatrix, traceless_part_slices, trace_part, rs, args...)
    num_chunks = length(rs) ÷ size(buffers, 2)

    traceless_part_chunks = Iterators.partition(traceless_part_slices, num_chunks)
    trace_part_chunks = Iterators.partition(trace_part, num_chunks)
    rs_chunks = Iterators.partition(rs, num_chunks)

    tasks = map(zip(eachslice(buffers, dims=2), traceless_part_chunks, trace_part_chunks, rs_chunks)) do chunk
        @spawn get_decomposition!(chunk..., args...)
    end

    fetch.(tasks)
    nothing
end
##
x = 1:400.0f0
y = 1:400.0f0
rs = Iterators.product(x, y)
buffer = Vector{ComplexF32}(undef, 6)
buffers = Matrix{ComplexF32}(undef, 6, 128)
trace_part = zeros(Float32, length(x) * length(y))
traceless_part = zeros(Float32, length(x) * length(y), 35)
traceles_part_slices = eachslice(traceless_part, dims=1)

get_decomposition!(buffer, traceles_part_slices, trace_part, rs, 200.0f0, 200.0f0, 50.0f0)
get_decomposition!(buffers, traceles_part_slices, trace_part, rs, 200.0f0, 200.0f0, 50.0f0)

@benchmark get_decomposition!($buffer, $traceles_part_slices, $trace_part, $rs, 200.0f0, 200.0f0, 50.0f0)
@benchmark get_decomposition!($buffers, $traceles_part_slices, $trace_part, $rs, 200.0f0, 200.0f0, 50.0f0)