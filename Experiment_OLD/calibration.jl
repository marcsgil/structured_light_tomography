using SpatialLightModulator
includet("ximea.jl")

function binary_mask(x, y, value)
    mask = zeros(UInt8, length(x), length(y))

    for j in 1:length(y)
        for i in 1:length(x)÷2
            mask[i, j] = value
        end
    end

    mask
end

slm = SLM()
##


width = 15.36f0
height = 8.64f0
resX = 1920
resY = 1080

x = LinRange(-width / 2, width / 2, resX)
y = LinRange(-height / 2, height / 2, resY)
##
holo = binary_mask(x,y, 0)
update_hologram(slm, holo)