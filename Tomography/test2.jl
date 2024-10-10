using QuantumMeasurements, AllocCheck

include("../Utils/basis.jl")
##
x = Float32.(1:224)
y = Float32.(1:224)
rs = Iterators.product(x, y)

buffer = Matrix{ComplexF32}(undef, 6, 512)
pars = (1.0f0, 200.0f0, 200.0f0, 50.0f0)

f!(buffer, r, pars) = fixed_order_basis!(buffer, r, pars)
g!(buffer, r, pars) = fixed_order_basis!(buffer, r, pars, Float32(π) / 2)
##
μ = empty_measurement(2 * 400^2, 6, Matrix{Float32})
μ1 = view(μ, 1:400^2, :)
μ2 = view(μ, 400^2+1:2*400^2, :)

multithreaded_update_measurement!(μ1, buffer, rs, pars, f!)
multithreaded_update_measurement!(μ2, buffer, rs, pars, g!)

@benchmark multithreaded_update_measurement!($μ1, $buffer, $rs, $pars, $f!)
@benchmark multithreaded_update_measurement!($μ2, $buffer, $rs, $pars, $g!)
##
itr = Iterators.map(r -> positive_l_basis!(buffer, r, pars), rs)


update_measurement!(μ, itr)

maximum(μ)

@benchmark update_measurement!($μ, $itr)

@benchmark update_measurement!($μ, $buffer, $rs, $pars, $positive_l_basis!)
##
using PositionMeasurements

position_measurement_matrix!(buffer, μ, rs, pars, basis!)