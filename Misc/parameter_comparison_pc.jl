using HDF5, LuxUtils, LuxCUDA
include("../Utils/model_fitting.jl")
include("../Utils/photocount_utils.jl")
includet("../Utils/ml_utils.jl")

model = get_model()
ps, st = jldopen("Tomography/TrainingLogs/best_model_pc.jld2") do file
    file["parameters"], file["states"]
end |> gpu_device()
##
path = "Data/Raw/fixed_order_photocount.h5"
calibration = h5open(path) do file
    file["calibration"] |> read
end

x = LinRange(-0.5f0, 0.5f0, size(calibration, 1))
y = LinRange(-0.5f0, 0.5f0, size(calibration, 2))

fit_d, fit_c = calibration_fit(x, y, calibration)

fit_c.param
##
orders = 1:4

pars = Matrix{Float64}(undef, 3, length(orders))
pars_std = Matrix{Float64}(undef, 3, length(orders))


for (n, order) ∈ enumerate(orders)
    x = h5open(path) do f
            f["images_order$order"][:, :, 2:2, :]
        end .|> Float32 |> gpu_device()
    normalize_data!(x, (1, 2))

    pred_pars = model(x, ps, st)[1]

    pars[:, n] = mean(pred_pars, dims=2) |> cpu_device()
    pars_std[:, n] = std(pred_pars, dims=2) |> cpu_device()
end

pars