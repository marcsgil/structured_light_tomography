using SpatialLightModulator, StructuredLight, HDF5, CairoMakie
import SpatialLightModulator: centralized_cut

includet("AcquisitionUtils/ximea.jl")
includet("AcquisitionUtils/capture_func.jl")
includet("../Utils/basis.jl")

slm = SLM()
##
width = 15.36f0
height = 8.64f0
resX = 1920
resY = 1080
w = 0.2f0
max_modulation = 82
x_period = 4
y_period = 3

X = LinRange(-width / 2, width / 2, resX)
Y = LinRange(-height / 2, height / 2, resY)
x = centralized_cut(X, 300)
y = centralized_cut(Y, 300)

incoming = hg(x, y, w=2.3f0)

config = (; width, height, resX, resY, max_modulation, x_period, y_period, incoming, x, y)
##
desired = lg(x, y; w, l=2)
holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period)
update_hologram!(slm, holo)
##
camera = XimeaCamera(
    "downsampling" => "XI_DWN_2x2",
    "width" => 200,
    "height" => 200,
    "offsetX" => 60,
    "offsetY" => 232,
    "exposure" => 15,
)
##
saving_path = "../Data/Raw/test_julia.h5"

n_masks = 200

ρs = h5open("../Data/template.h5") do file
    file["labels_dim2"][:, :, 1:100]
end
##
prompt_calibration(saving_path, w, camera, slm, config)

prompt_iris_measurement(saving_path, ρs, n_masks, w, camera, slm, config; sleep_time=0.05)
##
h5open(saving_path) do file
    @show keys(file)
end