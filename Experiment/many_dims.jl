using SpatialLightModulator, StructuredLight, HDF5, CairoMakie
import SpatialLightModulator: centralized_cut

includet("AcquisitionUtils/ximea.jl")
includet("AcquisitionUtils/capture_func.jl")
includet("../Utils/basis.jl")
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
desired = hg(x, y; w, m=8, n=8)
holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period)
update_hologram!(slm, holo)
##
camera = XimeaCamera(
    "downsampling" => "XI_DWN_2x2",
    "width" => 200,
    "height" => 200,
    "offsetX" => 40,
    "offsetY" => 288,
    "exposure" => 15,
)
##
visualize(capture(camera))
##
saving_path = "../Data/Raw/postive_l_new.h5"

n_masks = 200
##
prompt_calibration(saving_path, w, camera, slm, config)
##
for dim ∈ 2:6
    @info "Starting with dim = $dim"
    basis_functions = positive_l_basis(dim, [0, 0, w, 1])
    ρs = h5open("../Data/template.h5") do file
        file["labels_dim$dim"][:, :, 1:100]
    end
    save_basis_loop(saving_path, "images_dim$dim", basis_functions, ρs, n_masks, camera, slm, config; sleep_time=0.05)
end
##
h5open(saving_path) do file
    @show keys(file)
end