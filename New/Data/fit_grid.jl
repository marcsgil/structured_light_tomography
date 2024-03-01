using Optim, Tullio, LinearAlgebra

function loss(pars, gaussian)
    xmin = pars[1]
    xmax = pars[2]
    ymin = pars[3]
    ymax = pars[4]
    x = LinRange(xmin, xmax, size(gaussian, 1))
    y = LinRange(ymin, ymax, size(gaussian, 2))
    @tullio mse := (exp(-x[i]^2 - y[j]^2) - gaussian[i, j])^2
end

function get_limits(calibration, pars0=[-4.0, 4.0, -4.0, 4.0])
    direct_calib = normalize(calibration[:, :, 1], Inf)
    converted_calib = normalize(calibration[:, :, 2], Inf)

    (optimize(pars -> loss(pars, direct_calib), pars0).minimizer,
        optimize(pars -> loss(pars, converted_calib), pars0).minimizer)
end

function get_grid(limits, size)
    x = LinRange(limits[1], limits[2], size[1])
    y = LinRange(limits[3], limits[4], size[2])
    x, y
end