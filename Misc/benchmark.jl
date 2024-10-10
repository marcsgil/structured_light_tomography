using QuantumMeasurements

includet("../Utils/basis.jl")

x = Float32.(1:224)
y = Float32.(1:224)
rs = Iterators.product(x, y)
param = (1.0f0, 200.0f0, 200.0f0, 50.0f0)

μ = empty_measurement(length(rs), 6, Matrix{Float32})
μ2 = empty_measurement(length(rs), 6, Matrix{Float32})
buffer = Vector{ComplexF32}(undef, 6)

update_measurement!(μ, buffer, rs, param, positive_l_basis!)

buffers = Matrix{ComplexF32}(undef, 6, 512)

multithreaded_update_measurement!(μ2, buffers, rs, param, positive_l_basis!)

mthd = PreAllocatedLinearInversion(μ)

freqs = rand(Float32, length(rs))

estimate_state(freqs, μ, mthd)
##
@benchmark update_measurement!($μ, $buffer, $rs, $param, $positive_l_basis!)
@benchmark multithreaded_update_measurement!($μ, $buffers, $rs, $param, $positive_l_basis!)
@benchmark estimate_state($freqs, $μ, $mthd)
##