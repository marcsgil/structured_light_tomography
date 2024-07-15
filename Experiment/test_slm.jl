using SpatialLightModulator, StructuredLight

slm = SLM()

width = 15.36
height = 8.64
resX = 1920
resY = 1080

x = LinRange(-width / 2, width / 2, resX)
y = LinRange(-height / 2, height / 2, resY)

incoming = hg(x, y, w=2.4)
desired = hg(x, y, w=0.5, m=10,n=10)

holo = generate_hologram(desired, incoming, x, y, 82, 4, 5)'
update_hologram(slm, holo)