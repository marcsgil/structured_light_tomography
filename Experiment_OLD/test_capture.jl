using SpatialLightModulator, StructuredLight
slm = SLM()
##
width = 15.36f0
height = 8.64f0
resX = 1920
resY = 1080
w = 0.2
max_modulation = 82
x_period = 5
y_period = 4

x = LinRange(-width / 2, width / 2, resX)
y = LinRange(-height / 2, height / 2, resY)
#x = centralized_cut(X, 600)
#y = centralized_cut(Y, 600)

incoming = hg(x, y, w=2.3f0)
##
desired = lg(x, y; w, p=1)
holo = generate_hologram(desired, incoming, x, y,
    max_modulation, x_period, y_period)
update_hologram(slm, holo)
##
includet("ximea.jl")
camera = XimeaCamera(
    "downsampling" => "XI_DWN_2x2",
    "width" => 200,
    "height" => 200,
    "offsetX" => 56,
    "offsetY" => 36,
    "exposure" => 2000,
)
using CairoMakie
##

ls = 0:10

output = Array{UInt8,3}(undef, 200, 200, length(ls))

desireds = stack(lg(x, y; γ, l=l) for l ∈ ls)

loop_capture!(output, desireds, incoming, slm, camera,
    x, y, max_modulation, 5, 4; sleep_time=0.15)

basis_functions = positive_l_basis(2, [0, 0, 0.1, 1])
ρs = 

basis_loop(basis_functions, ρs, n_masks,
    saving_path, saving_name, incoming,
    slm, camera, x, y, max_modulation, 5, 4; sleep_time=0.15, par=nothing)