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

function scatter_and_errorbars!(ax, x, y, error, multiplier=1;
    color,
    marker,
)
    errorbars!(ax, x, y * multiplier, error * multiplier; color, whiskerwidth=12, linewidth=2)
    scatter!(ax, x, y * multiplier; color, marker, markersize=20)
end

get_magnitude(x) = round(Int, log10(x))

function make_plot(; waist_multiplier, waist, waist_std, waist_fit,
    position_multiplier, x, x_std, x_fit, y, y_std, y_fit, saving_path="",
    horizontal_values, horizontal_label)
    with_theme(theme_latexfonts()) do
        fig = Figure(fontsize=20)

        mag = get_magnitude(waist_multiplier)
        ax1 = CairoMakie.Axis(fig[1, 1],
            xlabel=horizontal_label,
            ylabel=L"w \ \left( \times 10^{-%$mag} \right)",
        )

        s1 = scatter_and_errorbars!(ax1, horizontal_values, waist, waist_std, waist_multiplier;
            color=:green,
            marker=:rect,
        )

        l1 = hlines!(ax1, waist_fit * waist_multiplier;
            color=:green,
            linestyle=:dash,
            linewidth=3)

        hidexdecorations!(ax1, grid=false)

        mag = get_magnitude(position_multiplier)
        ax2 = CairoMakie.Axis(fig[2, 1],
            xlabel=horizontal_label,
            ylabel=L"\mathbf{r}_0 \ \left(\times 10^{-%$mag}\right)",
        )

        s2 = scatter_and_errorbars!(ax2, horizontal_values, x, x_std, position_multiplier;
            color=:red,
            marker=:circle,
        )

        l2 = hlines!(ax2, x_fit * position_multiplier;
            color=:red,
            linestyle=:dot,
            linewidth=3)

        s3 = scatter_and_errorbars!(ax2, horizontal_values, y, y_std, position_multiplier;
            color=:blue,
            marker=:diamond,
        )

        l3 = hlines!(ax2, y_fit * position_multiplier;
            color=:blue,
            linestyle=:dashdot,
            linewidth=3)

        Legend(fig[:, 2], [s1, l1, s2, l2, s3, l3],
            [
                L"w^{NN}",
                L"w^{LS}",
                L"x^{NN}_0",
                L"x_0^{LS}",
                L"y^{NN}_0",
                L"y_0^{LS}",
            ])

        rowgap!(fig.layout, 5)

        if !isempty(saving_path)
            save(saving_path, fig)
        end

        fig
    end
end

relu(x::T1, y::T2) where {T1,T2} = x > y ? x - y : zero(promote_type(T1, T2))

function get_formated_data(pars, pars_std, fit)
    (
        waist=pars[3, :],
        waist_std=pars_std[3, :],
        waist_fit=fit.param[3],
        x=pars[1, :],
        x_std=pars_std[1, :],
        x_fit=fit.param[1],
        y=pars[2, :],
        y_std=pars_std[2, :],
        y_fit=fit.param[2]
    )
end
##
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


make_plot(; waist_multiplier=10^2, position_multiplier=10^2, saving_path="",
    horizontal_values=dims, horizontal_label="Dimension",
    get_formated_data(pars, pars_std, fit)...)
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

path = "Data/Raw/fixed_order_intense.h5"

calibration = h5open(path) do file
    read(file["calibration"])
end

x = LinRange(-0.5, 0.5, size(calibration, 1))
y = LinRange(-0.5, 0.5, size(calibration, 2))

p0 = Float64.([0, 0, 0.1, maximum(calibration), minimum(calibration)])

fit_d = surface_fit(gaussian_model, x, y, calibration[:, :, 1], p0)
fit_c = surface_fit(gaussian_model, x, y, calibration[:, :, 2], p0)

orders = 1:5
pars = Matrix{Float64}(undef, 3, length(dims))
pars_std = Matrix{Float64}(undef, 3, length(dims))
##
converted = 1

if converted == 1
    fit = fit_d
else
    fit = fit_c
end

name = converted == 1 ? "direct" : "converted"

conf_int = confidence_interval(fit, 0.05)
error = map(x -> (x[2] - x[1]) / 2, conf_int)

for (n, order) ∈ enumerate(orders)
    x = h5open("Data/Raw/fixed_order_intense.h5") do f
        imresize(f["images_order$order"][:, :, converted, :], 64, 64)
    end |> gpu_device()
    normalize_data!(x, (1, 2))
    x = reshape(x, 64, 64, 1, 100)

    pred_pars = model(x, ps, st)[1]

    pars[:, n] = mean(pred_pars, dims=2) |> cpu_device()
    pars_std[:, n] = std(pred_pars, dims=2) |> cpu_device()
end

make_plot(; waist_multiplier=10^2, position_multiplier=10^2, saving_path="",
    horizontal_values=orders, horizontal_label="Order",
    get_formated_data(pars, pars_std, fit)...)