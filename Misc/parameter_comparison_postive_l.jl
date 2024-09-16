using BayesianTomography, HDF5, ProgressMeter, LinearAlgebra, LuxUtils, LuxCUDA, Images
using CairoMakie
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/model_fitting.jl")
includet("../Utils/ml_utils.jl")

model = get_model()
ps, st = jldopen("Tomography/TrainingLogs/best_model.jld2") do file
    file["parameters"], file["states"]
end |> gpu_device()
##
relu(x::T1, y::T2) where {T1,T2} = x > y ? x - y : zero(promote_type(T1, T2))

function load_data(path, key)
    images, ρs = h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"]
    end

    map!(x -> relu(x, bg), images, images)

    images, ρs
end

path = "Data/Raw/positive_l.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, maximum(calibration), minimum(calibration)])
fit = surface_fit(gaussian_model, x, y, calibration, p0)

fit.param
conf_int = confidence_interval(fit, 0.05)
error = map(x -> (x[2] - x[1]) / 2, conf_int)

dims = 2:6

pars = Matrix{Float64}(undef, 3, length(dims))
pars_std = Matrix{Float64}(undef, 3, length(dims))

for (n, dim) ∈ enumerate(dims)
    x = h5open("Data/Raw/positive_l.h5") do f
        imresize(read(f["images_dim$dim"]), 64, 64)
    end |> gpu_device()
    normalize_data!(x, (1, 2))
    x = reshape(x, 64, 64, 1, 100)

    pred_pars = model(x, ps, st)[1]

    pars[:, n] = mean(pred_pars, dims=2) |> cpu_device()
    pars_std[:, n] = std(pred_pars, dims=2) |> cpu_device()
end
relative_waist_error = @. abs.(pars[3, :] / fit.param[3] - 1) * 100
relative_waist_error_std = @. pars_std[3, :] / fit.param[3] * 100
x_difference = abs.(pars[1, :] .- fit.param[1])
y_difference = abs.(pars[2, :] .- fit.param[2])
x_difference_std = pars_std[1, :]
y_difference_std = pars_std[2, :]

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)

    ax1 = CairoMakie.Axis(fig[1, 1],
        xlabel="Dimension",
        ylabel=L"|\delta w| \ \left( \times 10^{-2} \right)",
        yticks=0:2:10,
    )
    ylims!(ax1, 0, 10)

    scatter!(ax1, dims, relative_waist_error,
        color=:green,
        markersize=16,
        marker=:ltriangle,)
    errorbars!(ax1, dims, relative_waist_error, relative_waist_error_std,
        color=:green,
        whiskerwidth=10)

    text!(L"w^{LS} = %$(round(fit.param[3], sigdigits=4)) \pm %$(round(error[1], digits=4))",
        position=(0.06, 0.82), fontsize=20, space=:relative)

    hidexdecorations!(ax1, grid=false)

    ax2 = CairoMakie.Axis(fig[2, 1],
        xlabel="Dimension",
        ylabel=L"|\Delta \mathbf{r}_0| \ \left(\times 10^{-3}\right)",
    )

    scatter!(ax2, dims, x_difference * 10^3,
        color=:red,
        markersize=16,
        label=L"\Delta x_0^{NN}")
    errorbars!(ax2, dims, x_difference * 10^3, x_difference_std * 10^3,
        color=:red,
        whiskerwidth=10)

    scatter!(ax2, dims, y_difference * 10^3,
        color=:blue,
        markersize=16,
        label=L"\Delta y_0^{NN}",
        marker=:diamond)
    errorbars!(ax2, dims, y_difference * 10^3, y_difference_std * 10^3,
        color=:blue,
        whiskerwidth=10)

    text!(L"x_0^{LS} = (%$(round(10^3 * fit.param[1], digits=2)) \pm %$(round(10^3 * error[1], sigdigits=1))) \times 10^{-3}",
        position=(0.2, 0.8), fontsize=20, space=:relative)
    text!(L"y_0^{LS} = (%$(round(10^3 * fit.param[2], digits=2)) \pm %$(round(10^3 * error[2], sigdigits=1))) \times 10^{-3}",
        position=(0.2, 0.65), fontsize=20, space=:relative)


    axislegend(ax2, position=:lt, fontsize=24)

    rowgap!(fig.layout, 5)

    #save("Plots/positive_l_param.pdf", fig)

    fig
end
##
function load_data(path, order, bgs)
    images, ρs = h5open(path) do file
        Float32.(read(file["images_order$order"])), conj.(read(file["labels_order$order"]))
    end

    Threads.@threads for J ∈ eachindex(IndexCartesian(), images)
        images[J] = relu(images[J], bgs[J[3]])
    end

    for slice ∈ eachslice(images, dims=(3, 4))
        normalize!(slice, 1)
    end
    images ./= 2

    images, ρs
end
##
path = "Data/Raw/fixed_order_intense.h5"

calibration = h5open(path) do file
    read(file["calibration"])
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, maximum(calibration), minimum(calibration)])

fit_d = surface_fit(gaussian_model, x, y, calibration[:, :, 1], p0)
fit_c = surface_fit(gaussian_model, x, y, calibration[:, :, 2], p0)

conf_int = confidence_interval(fit_d, 0.05)
error = map(x -> (x[2] - x[1]) / 2, conf_int)

orders = 1:5
pars = Matrix{Float64}(undef, 3, length(dims))
pars_std = Matrix{Float64}(undef, 3, length(dims))

for (n, order) ∈ enumerate(orders)
    x = h5open("Data/Raw/fixed_order_intense.h5") do f
        imresize(f["images_order$order"][:, :, 1, :], 64, 64)
    end |> gpu_device()
    normalize_data!(x, (1, 2))
    x = reshape(x, 64, 64, 1, 100)

    pred_pars = model(x, ps, st)[1]

    pars[:, n] = mean(pred_pars, dims=2) |> cpu_device()
    pars_std[:, n] = std(pred_pars, dims=2) |> cpu_device()
end
relative_waist_error = @. abs.(pars[3, :] / fit.param[3] - 1) * 100
relative_waist_error_std = @. pars_std[3, :] / fit.param[3] * 100
x_difference = abs.(pars[1, :] .- fit.param[1])
y_difference = abs.(pars[2, :] .- fit.param[2])
x_difference_std = pars_std[1, :]
y_difference_std = pars_std[2, :]

with_theme(theme_latexfonts()) do
    fig = Figure(fontsize=20)

    ax1 = CairoMakie.Axis(fig[1, 1],
        xlabel="Dimension",
        ylabel=L"|\delta w| \ \left( \times 10^{-2} \right)",
        yticks=0:2:10,
    )
    ylims!(ax1, 0, 10)

    scatter!(ax1, dims, relative_waist_error,
        color=:green,
        markersize=16,
        marker=:ltriangle,)
    errorbars!(ax1, dims, relative_waist_error, relative_waist_error_std,
        color=:green,
        whiskerwidth=10)

    text!(L"w^{LS} = %$(round(fit.param[3], sigdigits=4)) \pm %$(round(error[1], digits=4))",
        position=(0.06, 0.82), fontsize=20, space=:relative)

    hidexdecorations!(ax1, grid=false)

    ax2 = CairoMakie.Axis(fig[2, 1],
        xlabel="Dimension",
        ylabel=L"|\Delta \mathbf{r}_0| \ \left(\times 10^{-3}\right)",
    )

    scatter!(ax2, dims, x_difference * 10^3,
        color=:red,
        markersize=16,
        label=L"\Delta x_0^{NN}")
    errorbars!(ax2, dims, x_difference * 10^3, x_difference_std * 10^3,
        color=:red,
        whiskerwidth=10)

    scatter!(ax2, dims, y_difference * 10^3,
        color=:blue,
        markersize=16,
        label=L"\Delta y_0^{NN}",
        marker=:diamond)
    errorbars!(ax2, dims, y_difference * 10^3, y_difference_std * 10^3,
        color=:blue,
        whiskerwidth=10)

    text!(L"x_0^{LS} = (%$(round(10^3 * fit.param[1], digits=2)) \pm %$(round(10^3 * error[1], sigdigits=1))) \times 10^{-3}",
        position=(0.2, 0.8), fontsize=20, space=:relative)
    text!(L"y_0^{LS} = (%$(round(10^3 * fit.param[2], digits=2)) \pm %$(round(10^3 * error[2], sigdigits=1))) \times 10^{-3}",
        position=(0.2, 0.65), fontsize=20, space=:relative)


    axislegend(ax2, position=:lt, fontsize=24)

    rowgap!(fig.layout, 5)

    #save("Plots/positive_l_param.pdf", fig)

    fig
end