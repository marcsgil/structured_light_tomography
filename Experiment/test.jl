using SpatialLightModulator, StructuredLight, HDF5

includet("AcquisitionUtils/capture_func.jl")
includet("../Utils/basis.jl")

slm = SLM()
##
width = 15.36f0
height = 8.64f0
resX = 1920
resY = 1080
w = 0.3
max_modulation = 82
x_period = 4
y_period = 3

x = LinRange(-width / 2, width / 2, resX)
y = LinRange(-height / 2, height / 2, resY)
#x = centralized_cut(X, 300)
#y = centralized_cut(Y, 300)

incoming = hg(x, y, w=2.3f0)

config = (; width, height, resX, resY, max_modulation, x_period, y_period, incoming, x, y)
##
desired = lg(x, y; w, l=2)
holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period, Simple)
update_hologram(slm, holo)
##
includet("AcquisitionUtils/ximea.jl")
camera = XimeaCamera(
    "downsampling" => "XI_DWN_2x2",
    "width" => 400,
    "height" => 400,
    "offsetX" => 164,
    "offsetY" => 0,
    "exposure" => 4000,
)
using CairoMakie
##
saving_path = "../Data/Raw/test_simple.h5"

n_masks = 300

ρs = h5open("../Data/template.h5") do file
    file["labels_dim2"][:, :, 1:3]
end

basis_functions = positive_l_basis(2, [0, 0, w, 1])
##
desired = lg(x, y; w)
holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period, Simple)
update_hologram(slm, holo)
calibration = capture(camera)
visualize(calibration) |> display
##
h5open(saving_path, "cw") do file
    file["calibration"] = calibration
end
##
save_basis_loop(saving_path, "image_1",
    basis_functions, ρs, n_masks, camera, slm, config;
    sleep_time=0.05)