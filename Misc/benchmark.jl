using BayesianTomography

includet("../Utils/position_operators.jl")
includet("../Utils/basis.jl")

x = Float32.(1:400)
y = Float32.(1:400)
param = Float32[200, 200, 50]

basis_d = fixed_order_basis(5, param)
basis_c = fixed_order_basis(5, param, -Float32(π) / 6)

operators = assemble_position_operators(x, y, basis_d, basis_c)
measurement = Measurement(operators)

mthd = LinearInversion()

freqs = rand(Float32, length(x) * length(y) * 2)

prediction(freqs, measurement, mthd)
##
@benchmark assemble_position_operators($x, $y, $basis_d, $basis_c)
@benchmark Measurement($operators)
@benchmark prediction($freqs, $measurement, $mthd)
##
mthd_ne = NormalEquations(measurement)
@benchmark NormalEquations($measurement)

prediction(freqs, measurement, mthd_ne)[1] ≈ prediction(freqs, measurement, mthd)[1]

@benchmark prediction($freqs, $measurement, $mthd_ne)