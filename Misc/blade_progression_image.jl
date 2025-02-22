using HDF5, CairoMakie
imgs = Matrix{UInt8}[]
push!(imgs,
    h5open("Data/positive_l.h5") do file
        file["images_dim2"][:, :, 1]
    end
)

for n ∈ (1, 2, 3)
    push!(imgs,
        h5open("Data/blade.h5") do file
            file["images_$n"][:, :, 1]
        end
    )
end

titles = ["x = $(x[1]); F = $(x[2])%" for x ∈ (("Inf", 99), ("-0.5w", 99), ("-1w", 98), ("-1.5w", 65))]
Js = (
    100-70:100+70,
    100-70:100+70,
    100-70:100+70,
    100-70:100+70
)


with_theme(theme_latexfonts()) do
    fig = Figure(; size = (1400, 400), fontsize=30)
    for n ∈ eachindex(imgs)
        ax = Axis(fig[1, n], aspect=DataAspect(), title = titles[n])
        heatmap!(ax, imgs[n][Js[n], Js[n]], colormap=:jet)
        hidedecorations!(ax)
    end
    #save("/Users/marcsgil/Code/Presentation-Tomography-SL/Images/blade_progression.pdf", fig)
    fig
end