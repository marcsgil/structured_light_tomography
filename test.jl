using SLMControl, CairoMakie, StructuredLight

rs = LinRange(-3f0, 3f0, 1920)
##
desired = lg(rs, rs; l=1, w=0.5f0)
incoming = hg(rs, rs)

holo = generate_hologram(desired, incoming, rs, rs, 255, 30, 40)
visualize(holo, colormap=:Greys)

@benchmark generate_hologram($desired, $incoming, $rs, $rs, 255, 30, 40)