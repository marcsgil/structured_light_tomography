using HDF5, BayesianTomography, ProgressMeter, Images, LuxUtils, LuxCUDA

includet("../Utils/model_fitting.jl")
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/ml_utils.jl")

relu(x::T1, y::T2) where {T1,T2} = x > y ? x - y : zero(promote_type(T1, T2))

function load_data(path, order, bgs)
    images, ρs = h5open(path) do file
        Float32.(read(file["images_order$order"])), conj.(read(file["labels_order$order"]))
    end

    Threads.@threads for J ∈ eachindex(IndexCartesian(), images)
        images[J] = relu(images[J], bgs[J[3]])
    end

    for slice ∈ eachslice(images, dims=(3, 4))
        normalize!(slice, 1)
    end
    images ./= 2

    images, ρs
end

function get_mode_prediction(param_d, param_c, probs, xs, ys, order)
    basis_d = fixed_order_basis(order, param_d)
    basis_c = [(x, y) -> f(x, y) * cis(-(k - 1) * π / 6)
               for (k, f) ∈ enumerate(fixed_order_basis(order, param_c))]

    povm = stack(assemble_position_operators(xs, ys, basis) for basis ∈ (basis_d, basis_c))

    problem = StateTomographyProblem(povm)

    mthd = LinearInversion(problem)

    prediction(probs, mthd)[1]
end
##
path = "Data/Raw/fixed_order_intense.h5"
#rs = LinRange(-0.5f0, 0.5f0, 400)
xs = Base.OneTo(400)
ys = Base.OneTo(400)
orders = 1:5
"""model = get_model()
ps, st = jldopen("Tomography/TrainingLogs/best_model.jld2") do file
    file["parameters"], file["states"]
end |> gpu_device()"""

metrics = Matrix{Float64}(undef, length(orders), 100)

p = Progress(100 * length(orders))

for (m, order) ∈ enumerate(orders)
    images, ρs = load_data(path, order, (0x02, 0x02))

    #images_ml = reshape(imresize(images, 64, 64), 64, 64, 2, 100) |> gpu_device()
    #normalize_data!(images_ml, (1, 2))

    #param_d = model(view(images_ml, :, :, 1:1, :), ps, st)[1] |> cpu_device()
    #param_c = model(view(images_ml, :, :, 2:2, :), ps, st)[1] |> cpu_device()

    for n ∈ axes(images, 4)
        probs = @view images[:, :, :, n]
        #pred_ρ = get_mode_prediction(view(param_d, :, n), view(param_c, :, n), probs, rs, order)
        param_d = center_of_mass_and_waist(view(probs, :, :, 1), order)
        param_c = center_of_mass_and_waist(view(probs, :, :, 2), order)
        pred_ρ = get_mode_prediction(param_d, param_c, probs, xs, ys, order)
        ρ = @view ρs[:, :, n]
        metrics[m, n] = fidelity(ρ, pred_ρ)
        next!(p)
    end
end

finish!(p)

vec(mean(metrics, dims=2))
##
h5open("Results/Intense/fixed_order_with_ml.h5", "cw") do file
    file["mean_fid"] = mean(metrics, dims=2)
    file["std_fid"] = std(metrics, dims=2)
    file["orders"] = collect(orders)
end