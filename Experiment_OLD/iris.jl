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
w = 0.2f0
max_modulation = 82
x_period = 5
y_period = 4

X = LinRange(-width / 2, width / 2, resX)
Y = LinRange(-height / 2, height / 2, resY)
x = centralized_cut(X, 300)
y = centralized_cut(Y, 300)

incoming = hg(x, y, w=2.4f0)
##
slm = SLM()
##
camera = XimeaCamera(
    "downsampling" => "XI_DWN_2x2",
    "width" => 200,
    "height" => 200,
    "offsetX" => 100,
    "offsetY" => 98,
    "exposure" => 1000,
)
using CairoMakie
##
saving_path = "Data/Raw/iris.h5"
density_matrix_path = "Data/template.h5"
##
get_calibration(saving_path, incoming, x, y, w, max_modulation, x_period, y_period, camera, slm)

prompt_iris_measurement(saving_path, density_matrix_path, 300,
    incoming, x, y, w, max_modulation, x_period, y_period, camera, slm; sleep_time=0.03)