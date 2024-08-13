using StructuredLight, SpatialLightModulator

slm = SLM()
width = 15.36f0
height = 8.64f0
resX = 1920
resY = 1080
w = 0.3
max_modulation = 82
x_period = 4
y_period = 3

x = LinRange(-width / 2, width / 2, resX)
y = LinRange(-height / 2, height / 2, resY)
incoming = hg(x,y,w=2.3)
##
desired = lg(x,y; w, l=2)
holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period)
update_hologram!(slm, holo)
##
close(slm)
##
using CairoMakie
visualize(SpatialLightModulator.centralized_cut(holo, (400,400)), colormap=:greys)
