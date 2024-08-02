using SpatialLightModulator, StructuredLight, ProgressMeter, PartiallyCoherentSources
using HDF5, BayesianTomography, LinearAlgebra, Tullio
includet("ximea.jl")
includet("../Utils/basis.jl")
includet("../Utils/model_fitting.jl")
includet("../Utils/capture.jl")
##
width = 15.36f0
height = 8.64f0
resX = 1920
resY = 1080
γ = 0.15f0
max_modulation = 82
x_period = 5
y_period = 4

X = LinRange(-width / 2, width / 2, resX)
Y = LinRange(-height / 2, height / 2, resY)
x = centralized_cut(X, 300)
y = centralized_cut(Y, 300)

incoming = hg(x, y, γ=2.3f0)
slm = SLM()
##
camera = XimeaCamera(
    "downsampling" => "XI_DWN_2x2",
    "width" => 200,
    "height" => 200,
    "offsetX" => 16,
    "offsetY" => 4,
    "exposure" => 2000,
)
using CairoMakie
##
saving_path = "Data/Raw/blade.h5"
density_matrix_path = "Data/template.h5"
##
get_calibration(saving_path, incoming, x, y, γ, max_modulation, x_period, y_period, camera, slm)

prompt_blade_measurement(saving_path, density_matrix_path, 300,
    incoming, x, y, γ, max_modulation, x_period, y_period, camera, slm; sleep_time=0.03)
##

h5open(saving_path) do file
    #display(visualize(file["calibration"] |> read))
    #display(visualize(file["images_1"][:, :, 6]))
    #file["fit_param"] |> read
    @show keys(file)
end

imgs = h5open(saving_path) do file
    file["images_1"] |> read
end

h5open(saving_path, "r") do file
    delete_object(file, "images_1")
end