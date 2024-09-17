using LuxUtils, LuxCUDA, Images, HDF5
includet("../Utils/ml_utils.jl")

device = gpu_device()

x, y = h5open("Data/Training/center_and_waist.h5") do file
    read(file["x"]), read(file["y"])
end

normalize_data!(x, (1, 2))
print("Dataset size: $(size(x,4)) \n")

model = get_model()

ps, st = train(x, y, 100, model; device,
    model_saving_path="Tomography/TrainingLogs/best_model.jld2",
    logging_path="Tomography/TrainingLogs/log.csv", patience=20);
