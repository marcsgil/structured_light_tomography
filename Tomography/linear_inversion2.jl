using LsqFit, HDF5, StructuredLight, PositionMeasurements, CairoMakie, BayesianTomography
includet("../Utils/basis.jl")
includet("../Utils/model_fitting.jl")

file = h5open("Data/Raw/mixed_intense.h5")
calibration = read(file["calibration"])
close(file)

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

xy = hcat(([x, y] for x in x, y in y)...)

p0 = Float64.([maximum(calibration), 0, 0, 0.1, 1, 3])
fit_direct = LsqFit.curve_fit(twoD_Gaussian, xy, view(calibration, :, :, 1) |> vec, p0)
fit_converted = LsqFit.curve_fit(twoD_Gaussian, xy, view(calibration, :, :, 2) |> vec, p0)

direct_basis = fixed_order_basis(1, fit_direct.param[4], fit_direct.param[5])
converted_basis = fixed_order_basis(1, fit_converted.param[4], fit_converted.param[5])

direct_operators = assemble_position_operators(x, y, direct_basis)
converted_operators = assemble_position_operators(xc, yc, basis)
mode_converter = diagm([cis(-k * π / 6) for k ∈ 0:order])
unitary_transform!(converted_operators, mode_converter)
operators = compose_povm(direct_operators, converted_operators)
mthd = LinearInversion(operators)