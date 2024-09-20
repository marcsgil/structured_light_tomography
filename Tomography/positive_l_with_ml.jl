using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra, LuxUtils, LuxCUDA, Images
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/ml_utils.jl")
##

relu(x::T1, y::T2) where {T1,T2} = x > y ? x - y : zero(promote_type(T1, T2))
function load_data(path, key, bg)
    images, ρs = h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"]
    end

    map!(x -> relu(x, bg), images, images)

    images, ρs
end

path = "Data/Raw/positive_l.h5"

bg = 0x02
##
"""model = get_model()
ps, st = jldopen("Tomography/TrainingLogs/best_model.jld2") do file
    file["parameters"], file["states"]
end |> gpu_device()"""
##
dims = 2:6
#rs = LinRange(-0.5f0, 0.5f0, 200)
xs = Base.OneTo(200)
ys = Base.OneTo(200)

metrics = Matrix{Float64}(undef, 100, length(dims))

@showprogress for (n, dim) ∈ enumerate(dims)
    images, ρs = load_data(path, "images_dim$dim", bg)

    #images_ml = reshape(imresize(images, 64, 64), 64, 64, 1, 100) |> gpu_device()
    #normalize_data!(images_ml, (1, 2))

    #param = model(images_ml, ps, st)[1] |> cpu_device()

    Threads.@threads for m ∈ axes(images, 3)
        probs = view(images, :, :, m)

        basis = positive_l_basis(dim, center_of_mass_and_waist(probs, 2 * (dim - 1)))
        povm = assemble_position_operators(xs, ys, basis)
        problem = StateTomographyProblem(povm)
        mthd = LinearInversion(problem)

        ρ = ρs[:, :, m]
        pred_ρ, _ = prediction(probs, mthd)

        metrics[m, n] = fidelity(ρ, pred_ρ)
    end
end

mean(metrics, dims=1)
##

h5open("Results/Intense/positive_l_with_ml.h5", "cw") do file
    file["mean_fid"] = mean(metrics, dims=1)
    file["std_fid"] = std(metrics, dims=1)
    file["dims"] = collect(dims)
end