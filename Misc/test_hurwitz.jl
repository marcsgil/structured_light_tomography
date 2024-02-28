using BayesianTomography

angles = rand(8) * π

hurwitz_parametrization(angles)
BayesianTomography.hurwitz_parametrization2(angles)

cos(angles[] / 2)
##
@benchmark hurwitz_parametrization($angles)
@benchmark BayesianTomography.hurwitz_parametrization2($angles)






sc = sincos.(angles[1:4]) |> stack
s = sc[1, :]

cumprod(s)


cumprod!