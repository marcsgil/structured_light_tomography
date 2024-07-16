using SpatialLightModulator, StructuredLight, ProgressMeter
includet("ximea.jl")

slm = SLM()

width = 15.36
height = 8.64
resX = 1920
resY = 1080

x = LinRange(-width / 2, width / 2, resX)
y = LinRange(-height / 2, height / 2, resY)

incoming = hg(x, y, w=2.4)
desired = hg(x, y, w=0.3, m=10, n=0)

holo = generate_hologram(desired, incoming, x, y, 82, 5, 4, 300)
update_hologram(slm, holo)
##
width = 200
height = 200
camera = XimeaCamera()
set_param(camera, "downsampling", "XI_DWN_2x2")
set_param(camera, "width", width)
set_param(camera, "height", height)
set_param(camera, "offsetX", 260)
set_param(camera, "offsetY", 208)
get_param(camera, "framerate")
##
@benchmark capture($camera)
##
buffer = Matrix{UInt8}(undef, width, height)
capture!(buffer, camera)
@benchmark capture!($buffer, $camera)
visualize(buffer)
##
N = 10
roi = 300
desireds = [hg(x, y; w=0.3, n) for n ∈ 1:N]
desireds = vcat((desireds for _ ∈ 1:10)...)
images = Array{UInt8,3}(undef, width, height, length(desireds))

holo = generate_hologram(desireds[1], incoming, x, y, 82, 5, 4, roi)
@showprogress for n ∈ 1:length(desireds)-1
    update_hologram(slm, holo, sleep_time=0.1)
    holo = generate_hologram(desireds[n+1], incoming, x, y, 82, 5, 4, roi)
    capture!(view(images, :, :, n), camera)
end

update_hologram(slm, holo)
capture!(view(images, :, :, N), camera)

12 * 3 * 50 / 60
##
using CairoMakie
visualize(images[:, :, 6])

close(camera)