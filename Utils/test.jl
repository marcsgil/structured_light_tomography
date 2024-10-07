using StructuredLight, BayesianTomography

function positive_l_basis!(dest, x, y, x₀, y₀, w)
    dim = length(dest)
    for n ∈ eachindex(dest)
        dest[n] = lg(x - x₀, y - y₀; w, p=dim - 2 - n, l=2n - 2)
    end
end

function get_decomposition!(buffer, traceless_part, trace_part, rs, args...)
    for (n, r, slice) ∈ zip(eachindex(trace_part), rs, eachslice(traceless_part, dims=1))
        positive_l_basis!(buffer, r..., args...)
        gell_mann_projection!(slice, buffer)
        trace_part[n] = sum(abs2, buffer)
    end
end
##
x = 1:400.0f0
y = 1:400.0f0
rs = Iterators.product(x, y)
buffer = Vector{ComplexF32}(undef, 6)
trace_part = Vector{Float32}(undef, length(x) * length(y))
traceless_part = Matrix{Float32}(undef, length(x) * length(y), 35)

@benchmark get_decomposition!($buffer, $traceless_part, $trace_part, $rs, 200.0f0, 200.0f0, 50.0f0)

@code_warntype get_decomposition!(buffer, traceless_part, trace_part, rs, 200.0f0, 200.0f0, 50.0f0)