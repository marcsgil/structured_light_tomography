using QuantumMeasurements, StructuredLight

function positive_l_basis!(dest, r, pars)
    dim = length(dest)
    x, y = r
    sqrt_δA, x₀, y₀, w = pars
    for n ∈ eachindex(dest)
        p = dim - n
        l = 2 - 2n # We calculate the negative value of l to get the conjugate
        dest[n] = lg(x - x₀, y - y₀; w, p, l) * sqrt_δA
    end
    dest
end
##
x = 1:400
y = 1:400
rs = Iterators.product(x, y)

buffer = Vector{ComplexF32}(undef, 6)
pars = (1, 200, 200, 50)

μ = empty_measurement(400^2, 6, Matrix{Float32})
#μ = empty_measurement(400^2, 6, ProportionalMeasurement{Float32,Matrix{Float32}})
update_measurement!(μ, buffer, rs, pars, positive_l_basis!)

@benchmark update_measurement!($μ, $buffer, $rs, $pars, $positive_l_basis!)

itr = Iterators.map(r -> positive_l_basis!(buffer, r, pars), rs)


update_measurement!(μ, itr)

maximum(μ)

@benchmark update_measurement!($μ, $itr)

@benchmark update_measurement!($μ, $buffer, $rs, $pars, $positive_l_basis!)
##
using PositionMeasurements

position_measurement_matrix!(buffer, μ, rs, pars, basis!)