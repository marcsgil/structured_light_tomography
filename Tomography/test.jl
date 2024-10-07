using Base.Threads: nthreads, @spawn

tmap!(args...; tasks_per_thread=2) = _tmap!(args..., tasks_per_thread)

function _tmap!(f, dest::Array, itr, tasks_per_thread::Int)
    chunk_size = max(1, length(itr) ÷ (tasks_per_thread * nthreads()))

    dest_chunk = Iterators.partition(dest, chunk_size)
    itr_chunk = Iterators.partition(itr, chunk_size)

    map(dest_chunk, itr_chunk) do dest, itr
        @spawn map!(f, dest, itr)
    end .|> fetch

    dest
end

_tmap!(f, dest, itr::AbstractArray, ::Nothing) = map!(f, dest, itr)

function _tmap!(f, dest, itr, ::Nothing)
    next = iterate(itr)
    for n ∈ eachindex(dest)
        val, state = next
        dest[n] = f(val)
        next = iterate(itr, state)
        isnothing(next) && break
    end
    dest
end
##
itr = (i for i ∈ 1:10^6)
dest = Vector{Int}(undef, 10^6)
f(x) = x^2

tmap!(f, dest, itr, tasks_per_thread=nothing)

@which _tmap!(f, dest, itr, nothing)
##
@benchmark tmap!($f, $dest, $itr, tasks_per_thread=2)
@benchmark map!($f, $dest, $itr)
@benchmark tmap!2($f, $dest, $itr)
##
using CUDA

itr = LinRange(0, 1, 10^6) |> Array |> cu
dest = similar(itr)
f(x) = x^2

tmap!(x -> x^2, dest, itr)

@benchmark tmap!($f, $dest, $itr)
@benchmark map!($f, $dest, $itr)