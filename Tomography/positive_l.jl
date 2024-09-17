using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/model_fitting.jl")
##
relu(x::T1, y::T2) where {T1,T2} = x > y ? x - y : zero(promote_type(T1, T2))

function load_data(path, key)
    images, ρs = h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"]
    end

    map!(x -> relu(x, bg), images, images)

    images, ρs
end

path = "Data/Raw/positive_l.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, maximum(calibration), minimum(calibration)])
fit = surface_fit(gaussian_model, x, y, calibration, p0)

bg = round(UInt8, fit.param[5])

fit.param
##
dims = 2:6

metrics = Matrix{Float64}(undef, 100, length(dims))

@showprogress for (n, dim) ∈ enumerate(dims)
    images, ρs = load_data(path, "images_dim$dim")

    basis = positive_l_basis(dim, fit.param)
    povm = assemble_position_operators(x, y, basis)
    problem = StateTomographyProblem(povm)
    mthd = LinearInversion(problem)

    Threads.@threads for m ∈ axes(images, 3)
        probs = images[:, :, m]
        ρ = ρs[:, :, m]
        pred_ρ, _ = prediction(probs, mthd)

        metrics[m, n] = fidelity(ρ, pred_ρ)
    end
end

mean(metrics, dims=1)
##
h5open("Results/Intense/positive_l.h5", "cw") do file
    file["dims"] = collect(dims)
    file["mean_fid"] = mean(metrics, dims=1)
    file["std_fid"] = std(metrics, dims=1)
end