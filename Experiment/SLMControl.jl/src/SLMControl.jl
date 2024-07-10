module SLMControl

using Roots, Interpolations, Tullio, Bessels, LinearAlgebra

include("inverse_functions.jl")
export inverse_besselj1

include("holograms.jl")
export generate_hologram

end
