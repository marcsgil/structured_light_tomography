using SpatialLightModulator, StructuredLight, HDF5

includet("AcquisitionUtils/capture_func.jl")
includet("../Utils/basis.jl")

slm = SLM()
##
includet("AcquisitionUtils/ximea.jl")
camera = XimeaCamera(
    "downsampling" => "XI_DWN_2x2",
    "width" => 200,
    "height" => 200,
    "offsetX" => 192,
    "offsetY" => 114,
    "exposure" => 2000,
)
using CairoMakie
##
##
width = 15.36f0
height = 8.64f0
resX = 1920
resY = 1080
w = 0.2
max_modulation = 82
x_period = 5
y_period = 4

X = LinRange(-width / 2, width / 2, resX)
Y = LinRange(-height / 2, height / 2, resY)
x = centralized_cut(X, 300)
y = centralized_cut(Y, 300)

incoming = hg(x, y, w=2.3f0)

config = (; width, height, resX, resY, max_modulation, x_period, y_period, incoming, x, y)
##
desired = hg(x, y; w, m=4)
holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period)
update_hologram(slm, holo)

visualize(capture(camera))