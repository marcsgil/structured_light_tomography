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

titles = [L"F = (%$(x[1]) \pm  %$(x[2]))%" for x ∈ ((99.3, 0.1), (98.2, 0.2), (96.0, 0.8), (62, 3))]
texts = [L"r = %$x" for x ∈ ("∞", "0.7w", "0.5w", "0.4w")]

Js = (
    100-70:100+70,
    100-40:100+40,
    100-30:100+30,
    100-20:100+20
)


with_theme(theme_latexfonts()) do
    fig = Figure(; size = (1400, 400))
    for n ∈ eachindex(imgs)
        ax = Axis(fig[1, n], aspect=DataAspect(), title = titles[n], titlesize=24)
        heatmap!(ax, imgs[n][Js[n], Js[n]], colormap=:jet)
        text!(ax, 0.95, 0, text=texts[n], color=:orange, space=:relative, align=(:right, :bottom), fontsize=32)
        hidedecorations!(ax)
    end
    #save("/Users/marcsgil/Code/Structured light tomography from position measurements/iris_progression.pdf", fig)
    fig
end