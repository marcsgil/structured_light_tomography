using StructuredLight, BayesianTomography
using Base.Threads: @spawn

function positive_l_basis!(dest, x, y, pars)
    dim = length(dest)
    sqrt_δA, x₀, y₀, w = pars
    for n ∈ eachindex(dest)
        p = dim - n
        l = 2n - 2
        dest[n] = lg(x - x₀, y - y₀; w, p, l) * sqrt_δA
    end
end

function get_decomposition!(buffer, traceless_part_slices, trace_part, rs, basis!, pars)
    for (n, r, slice) ∈ zip(eachindex(trace_part), rs, traceless_part_slices)
        basis!(buffer, r[1], r[2], pars)
        gell_mann_projection!(slice, buffer)
        trace_part[n] = sum(abs2, buffer) / length(buffer)
    end
end

function get_decomposition!(buffers::AbstractMatrix, traceless_part_slices, trace_part, rs, basis!, pars)
    chunk_size = cld(length(rs), size(buffers, 2))

    traceless_part_chunks = Iterators.partition(traceless_part_slices, chunk_size)
    trace_part_chunks = Iterators.partition(trace_part, chunk_size)
    rs_chunks = Iterators.partition(rs, chunk_size)

    #@show size(buffers, 2)
    #@show length(trace_part_chunks)

    tasks = map(zip(eachslice(buffers, dims=2), traceless_part_chunks, trace_part_chunks, rs_chunks)) do chunk
        @spawn get_decomposition!(chunk..., basis!, pars)
    end

    fetch.(tasks)
    nothing
end

function set_position_measurement!(measurement, buffers, rs, basis!, pars)
    get_decomposition!(buffers, eachslice(measurement.traceless_part, dims=1),
        measurement.trace_part, rs, basis!, pars)
    nothing
end

function get_position_measurement(rs, dim, basis, pars; nbuffers::Int=128)
    T = float(eltype(first(rs)))
    trace_part = Vector{T}(undef, length(rs))
    traceless_part = Matrix{T}(undef, length(rs), dim^2 - 1)
    if nbuffers > 0
        buffers = Matrix{complex(T)}(undef, dim, nbuffers)
    else
        buffers = Vector{complex(T)}(undef, dim)
    end

    μ = Measurement(traceless_part, trace_part, dim)

    set_position_measurement!(μ, buffers, rs, basis, pars)

    μ
end
##
using BayesianTomography, HDF5, ProgressMeter
includet("model_fitting.jl")
##
function load_data(path, key, bg)
    images, ρs = h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"]
    end

    remove_background!(images, bg)

    images, ρs
end

path = "Data/positive_l.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = axes(calibration, 1)
y = axes(calibration, 2)
rs = Iterators.product(x, y)

fit = calibration_fit(x, y, calibration)

bg = round(UInt8, fit.param[5])

fit.param
##
dim = 6
images, ρs = load_data(path, "images_dim$dim", bg)

T = float(eltype(first(rs)))
trace_part = Vector{T}(undef, length(rs))
traceless_part = Matrix{T}(undef, length(rs), dim^2 - 1)
buffer = Vector{complex(T)}(undef, dim)
pars = (1, fit.param...)

μ = get_position_measurement(rs, dim, positive_l_basis!, pars; nbuffers=128)

sum(μ.trace_part), sum(trace_part)

#maximum(μ.traceless_part)

@benchmark get_position_measurement($rs, $dim, $positive_l_basis!, $pars)
##
dims = 2:6

fid = Matrix{Float64}(undef, 100, length(dims))

@belapsed for (n, dim) ∈ enumerate(dims)
    images, ρs = load_data(path, "images_dim$dim", bg)

    μ = get_position_measurement(rs, dim, positive_l_basis!, (1, fit.param...), nbuffers=128)
    mthd = PreAllocatedLinearInversion(μ)

    Threads.@threads for m ∈ axes(images, 3)
        probs = @view images[:, :, m]
        ρ = @view ρs[:, :, m]
        pred_ρ = prediction(probs, μ, mthd)[1]

        fid[m, n] = fidelity(ρ, pred_ρ)
    end
end

mean(fid, dims=1)
##

collect((x,y) for x ∈ x, y ∈ y if x^2 + y^2 ≤ 200^2)
##

ρ = BayesianTomography.sample(GinibreEnsamble(20))

gell_mann_projection2(ρ) ≈ gell_mann_projection(ρ)

gell_mann_projection2(ρ) = [real(ρ ⋅ ω) for ω ∈ GellMannMatrices(size(ρ, 1))]

@benchmark gell_mann_projection($ρ)
@benchmark gell_mann_projection2($ρ)