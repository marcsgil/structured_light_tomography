using SpatialLightModulator, StructuredLight
slm = SLM()
##
width = 15.36f0
height = 8.64f0
resX = 1920
resY = 1080
γ = 0.15f0
max_modulation = 82
x_period = 5
y_period = 4

x = LinRange(-width / 2, width / 2, resX)
y = LinRange(-height / 2, height / 2, resY)
#x = centralized_cut(X, 600)
#y = centralized_cut(Y, 600)

incoming = hg(x, y, γ=2.3f0)
##
x₀ = -2f0γ
desired = lg(x .-x₀, y; γ, p=1)
holo = generate_hologram(desired, incoming, x, y, 
max_modulation, x_period, y_period)
update_hologram(slm, holo)