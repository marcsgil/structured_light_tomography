using SpatialLightModulator, StructuredLight

using PythonCall
xiapi = pyimport("ximea.xiapi")
camera = xiapi.Camera()
camera.open_device()
camera.set_exposure(1000)
image = xiapi.Image()
camera.start_acquisition()

slm = SLM()

width = 15.36
height = 8.64
resX = 1920
resY = 1080

x = LinRange(-width / 2, width / 2, resX)
y = LinRange(-height / 2, height / 2, resY)

incoming = hg(x, y, w=2.4)
desired = hg(x, y, w=0.5, m=10, n=10)

holo = generate_hologram(desired, incoming, x, y, 82, 5, 4)'
update_hologram(slm, holo)

images = Array{UInt8,3}(undef, 1280, 1024, 11)

@benchmark hg($x, $y; w=0.3, n=10)

for n ∈ 0:size(images, 3)-1
    desired = hg(x, y; w=0.3, n)
    holo = generate_hologram(desired, incoming, x, y, 82, 5, 4)'
    update_hologram(slm, holo)
    sleep(0.2)
    camera.get_image(image)
    images[:, :, n+1] = pyconvert(Matrix{UInt8}, image.get_image_data_numpy())'
end

using CairoMakie
visualize(images[:, :, 11])

camera.stop_acquisition()
camera.close_device()