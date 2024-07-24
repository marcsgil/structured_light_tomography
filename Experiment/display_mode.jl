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
w = 0.3f0
max_modulation = 82
x_period = 4
y_period = 5

X = LinRange(-width / 2, width / 2, resX)
Y = LinRange(-height / 2, height / 2, resY)
x = centralized_cut(X, 600)
y = centralized_cut(Y, 600)

incoming = hg(x, y, w=2.4f0)
##
slm = SLM()
##
display_calibration(w, incoming, x, y, max_modulation, x_period, y_period, slm)
##
desired = lg(x, y; w, l = 2)
holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period)
update_hologram(slm, holo)