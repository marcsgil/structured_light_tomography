using HDF5, CairoMakie
imgs = Matrix{UInt8}[]
push!(imgs,
    h5open("Data/positive_l.h5") do file
        file["images_dim2"][:, :, 1]
    end
)

for n ∈ (3, 1, 2)
    push!(imgs,
        h5open("Data/iris.h5") do file
            file["images_$n"][:, :, 1]
        end
    )
end

titles = ["r = $(x[1]); F = $(x[2])%" for x ∈ (("Inf", 99), ("0.74w", 98), ("0.50w", 96), ("0.37w", 62))]
Js = (
    100-70:100+70,
    100-40:100+40,
    100-30:100+30,
    100-20:100+20
)


with_theme(theme_latexfonts()) do
    fig = Figure(; size = (1400, 400), fontsize=30)
    for n ∈ eachindex(imgs)
        ax = Axis(fig[1, n], aspect=DataAspect(), title = titles[n])
        heatmap!(ax, imgs[n][Js[n], Js[n]], colormap=:jet)
        hidedecorations!(ax)
    end
    #save("/Users/marcsgil/Code/Presentation-Tomography-SL/Images/iris_progression.pdf", fig)
    fig
end