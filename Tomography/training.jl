using LuxUtils, LuxCUDA, Images, HDF5
includet("../Utils/ml_utils.jl")

device = gpu_device()

x, y = h5open("Data/Training/center_and_waist.h5") do file
    read(file["x"]), read(file["y"])
end

normalize_data!(x, (1, 2))

model = get_model()

ps, st = train(x, y, 200, model; device,
    model_saving_path="Tomography/TrainingLogs/best_model.jld2",
    logging_path="Tomography/TrainingLogs/log.csv", patience=30);
##
"""x_exp = h5open("Data/Raw/positive_l.h5") do f
    imresize(read(f["images_dim2"]), 64, 64)
end |> device"""

x_exp = h5open("Data/Raw/fixed_order_intense.h5") do f
    imresize(f["images_order5"][:, :, 2, :], 64, 64)
end |> device

normalize_data!(x_exp, (1, 2))

x_exp = reshape(x_exp, 64, 64, 1, 100)

mean(model(x_exp, ps, st)[1], dims=2)
#std(model(x_exp, ps, st)[1], dims=2)