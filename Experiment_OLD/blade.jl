using SpatialLightModulator, StructuredLight, ProgressMeter, PartiallyCoherentSources
using HDF5, BayesianTomography, LinearAlgebra, Tullio
includet("ximea.jl")
includet("../Utils/basis.jl")
includet("../Utils/model_fitting.jl")
includet("../Utils/capture.jl")
slm = SLM()
##
width = 15.36f0
height = 8.64f0
resX = 1920
resY = 1080
γ = 0.15f0
max_modulation = 82
x_period = 5
y_period = 4

x = LinRange(-width / 2, width / 2, resX)
y = LinRange(-height / 2, height / 2, resY)
#x = centralized_cut(X, 300)
#y = centralized_cut(Y, 300)

incoming = hg(x, y, γ=1.6f0)
desired = lg(x, y; γ, p=1)
holo = generate_hologram(desired, incoming, x, y,
    max_modulation, x_period, y_period)
update_hologram(slm, holo)
##
camera = XimeaCamera(
    "downsampling" => "XI_DWN_2x2",
    "width" => 200,
    "height" => 200,
    "offsetX" => 204,
    "offsetY" => 54,
    "exposure" => 2000,
)
using CairoMakie
##
saving_path = "Data/Raw/blade.h5"
density_matrix_path = "Data/template.h5"
##
get_calibration(saving_path, incoming, x, y, γ, max_modulation, x_period, y_period, camera, slm)

prompt_blade_measurement(saving_path, density_matrix_path, 200,
    incoming, x, y, γ, max_modulation, x_period, y_period, camera, slm; sleep_time=0.15)
##

visualize(capture(camera))

h5open(saving_path) do file
    #display(visualize(file["calibration"] |> read))
    #display(visualize(file["images_1"][:, :, 6]))
    #file["fit_param"] |> read
    @show keys(file)
end
##
imgs = h5open(saving_path) do file
    file["images_1"] |> read
end

visualize(imgs[:, :, 3])
##
h5open(saving_path, "cw") do file
    delete_object(file, "images_1")
end