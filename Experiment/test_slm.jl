using SpatialLightModulator, StructuredLight, ProgressMeter, PartiallyCoherentSources
using HDF5, BayesianTomography, LinearAlgebra, Tullio
includet("ximea.jl")
includet("../basis.jl")

function loop_capture!(output, desireds, incoming, slm, camera,
    x, y, max_modulation, xperiod, yperiod; sleep_time=0.15)
    N = size(desireds, 3)
    holo = generate_hologram(view(desireds, :, :, 1), incoming, x, y, max_modulation, xperiod, yperiod)
    @showprogress for n ∈ 1:N-1
        update_hologram(slm, holo; sleep_time)
        holo = generate_hologram(view(desireds, :, :, n + 1), incoming, x, y, 82, 5, 4)
        capture!(view(output, :, :, n), camera)
    end

    update_hologram(slm, holo)
    capture!(view(output, :, :, N), camera)
    nothing
end

function basis_loop(n_modes, n_masks, basis_functions,
    saving_path, saving_name, incoming,
    slm, camera, x, y, max_modulation, xperiod, yperiod; sleep_time=0.15)

    file = h5open(saving_path, "cw")

    basis = stack(f(x, y) for f ∈ basis_functions)
    ρs = sample(GinibreEnsamble(size(basis, 3)), n_modes)
    eigen_states = similar(basis)
    desireds = Array{ComplexF32,3}(undef, size(basis, 1), size(basis, 2), n_masks)

    width = pyconvert(Int, camera.camera.get_param("width"))
    height = pyconvert(Int, camera.camera.get_param("height"))
    output = Array{UInt8,3}(undef, width, height, n_masks)

    result = Array{UInt8,3}(undef, width, height, n_modes)

    for n ∈ axes(ρs, 3)
        @info "Processing mode $n"
        ρ = view(ρs, :, :, n)
        F = eigen(Hermitian(ρ))
        @tullio eigen_states[j, k, n] = F.vectors[m, n] * basis[j, k, m]

        generate_fields!(desireds, eigen_states, F.values)
        loop_capture!(output, desireds, incoming, slm, camera, x, y, max_modulation, xperiod, yperiod; sleep_time)

        map!(x -> round(UInt8, x), view(result, :, :, n), mean(output, dims=3))
    end

    file["images_"*saving_name] = result
    file["labels_"*saving_name] = ρs
    close(file)
end

function full_loop(n_modes, n_masks, max_dims, w,
    saving_path, incoming,
    slm, camera, x, y, max_modulation, xperiod, yperiod; sleep_time=0.15)

    file = h5open(saving_path, "cw")
    desired = hg(x, y; w)
    holo = generate_hologram(desired, incoming, x, y, max_modulation, xperiod, yperiod)
    update_hologram(slm, holo)
    file["calibration"] = capture(camera)
    close(file)

    for dims ∈ 2:max_dims
        @info "Processing dim $dims"
        basis_functions = positive_l_basis(dims, w)
        basis_loop(n_modes, n_masks, basis_functions,
            saving_path, "dim$dims",
            incoming, slm, camera, x, y, max_modulation, xperiod, yperiod; sleep_time)
    end
end

slm = SLM()
##
width = 15.36f0
height = 8.64f0
resX = 1920
resY = 1080
w = 0.3f0

X = LinRange(-width / 2, width / 2, resX)
Y = LinRange(-height / 2, height / 2, resY)
x = centralized_cut(X, 300)
y = centralized_cut(Y, 300)

incoming = hg(x, y, w=2.4f0)
desired = hg(x, y; w, n=1)

holo = generate_hologram(desired, incoming, x, y, 82, 5, 4)
update_hologram(slm, holo, sleep_time=0)
##
width = 200
height = 200
camera = XimeaCamera()
set_param(camera, "downsampling", "XI_DWN_2x2")
set_param(camera, "width", width)
set_param(camera, "height", height)
set_param(camera, "offsetX", 28)
set_param(camera, "offsetY", 254)
set_param(camera, "exposure", 1000)
get_param(camera, "framerate")
##
@benchmark capture($camera)
##
using CairoMakie
buffer = Matrix{UInt8}(undef, width, height)
capture!(buffer, camera)

visualize(buffer)
##
basis_functions = positive_l_basis(2, w)
#basis_loop(2, 300, basis_functions, "test.h5", "order1", incoming, slm, camera, x, y, 82, 5, 4)

full_loop(100, 300, 6, w,
    "Data/Raw/positive_l.h5", incoming,
    slm, camera, x, y, 82, 5, 4; sleep_time=0.05)
##
file = h5open("Data/Raw/positive_l.h5")

dim = 6
n = 2
ρ = file["labels_dim$dim"][:, :, n]
basis_functions = positive_l_basis(dim, w)
basis = stack(f(x, y) for f ∈ basis_functions)

using CairoMakie
@tullio mean_theo[j, k] := ρ[m, n] * basis[j, k, m] * conj(basis[j, k, n]) |> real
visualize(mean_theo)
visualize(file["images_dim$dim"][:, :, n])

visualize(file["calibration"][:, :])

maximum(file["images_dim2"][:, :, n]) |> Int
file["images_order1"][:, :, 1]
close(file)


close(camera)