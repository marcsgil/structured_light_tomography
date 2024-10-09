using QuantumMeasurements, HDF5, ProgressMeter, PositionMeasurements
includet("../Utils/model_fitting.jl")
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

fit_result = calibration_fit(x, y, calibration)

bg = round(UInt8, fit_result.param[5])

fit_result.param
##
empty_measurement(400^2, 6, ProportionalMeasurement{Float32,Matrix{Float32}})
##
dims = 2:6
fid = Matrix{Float64}(undef, 100, length(dims))

@showprogress for (n, dim) ∈ enumerate(dims)
    images, ρs = load_data(path, "images_dim$dim", bg)
    pars = (1, fit_result.param...)

    buffer = Vector{ComplexF32}(undef, dim)
    μ = empty_measurement(200^2, dim, Matrix{Float32})
    update_measurement!(μ, buffer, rs, pars, positive_l_basis!)

    """itr = Iterators.map(rs) do r
        positive_l_basis!(buffer, r, pars)
        buffer
    end
    μ = ProportionalMeasurement(itr)"""

    mthd = LinearInversion()

    for m ∈ axes(images, 3)
        probs = @view images[:, :, m]
        ρ = @view ρs[:, :, m]
        pred_ρ = estimate_state(probs, μ, mthd)[1]

        fid[m, n] = fidelity(ρ, pred_ρ)
    end
end

mean(fid, dims=1)