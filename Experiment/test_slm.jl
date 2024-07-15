using SpatialLightModulator, StructuredLight


slm = SLM()

width = 15.36
height = 8.64
resX = 1920
resY = 1080

x = LinRange(-width / 2, width / 2, resX)
y = LinRange(-height / 2, height / 2, resY)

incoming = hg(x, y, w=2.4)
desired = lg(x, y, w=0.5, l=2)

holo = generate_hologram(desired, incoming, x, y, 82, 20, 30)'
update_hologram!(slm, holo)

for _ ∈ range(1, 100)
    update_hologram!(slm, rand(UInt8, slm.width, slm.height))
    sleep(1 / slm.framerate)
end