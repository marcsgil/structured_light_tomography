using SpatialLightModulator, StructuredLight, ProgressMeter, PartiallyCoherentSources
using HDF5, BayesianTomography, LinearAlgebra, Tullio
includet("ximea.jl")
includet("../Utils/basis.jl")
includet("../Utils/model_fitting.jl")
includet("../Utils/loops.jl")
##
width = 15.36f0
height = 8.64f0
resX = 1920
resY = 1080
w = 0.3f0
max_modulation = 82
x_period = 5
y_period = 4

X = LinRange(-width / 2, width / 2, resX)
Y = LinRange(-height / 2, height / 2, resY)
x = centralized_cut(X, 300)
y = centralized_cut(Y, 300)

incoming = hg(x, y, w=2.4f0)
slm = SLM()
##
desired = hg(x, y; w, n=10, m=10)

holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period)
update_hologram(slm, holo, sleep_time=0)
##
camera = XimeaCamera(
    "downsampling" => "XI_DWN_2x2",
    "width" => 200,
    "height" => 200,
    "offsetX" => 8,
    "offsetY" => 230,
    "exposure" => 1000,
)
using CairoMakie
##
function get_calibration(saving_path, x, y, w, max_modulation, x_period, y_period, camera, slm)
    desired = hg(x, y; w)

    holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period)
    update_hologram(slm, holo)

    while true
        calibration = capture(camera)
        display(visualize(calibration))

        println("Do you accept?")
        answer = readline()

        if answer == "y"
            @info "Calculating fit"
            x_cam = axes(calibration, 1)
            y_cam = axes(calibration, 2)
            xy = hcat(([x, y] for x in x_cam, y in y_cam)...)

            p0 = Float64.([length(x_cam) ÷ 2, length(y_cam) ÷ 2, length(x_cam) ÷ 10, 1, maximum(calibration), minimum(calibration)])
            fit = LsqFit.curve_fit(twoD_Gaussian, xy, calibration[:, :, 1] |> vec, p0)

            h5open(saving_path, "cw") do file
                file["x"] = x_cam |> collect
                file["y"] = y_cam |> collect
                file["fit_param"] = fit.param
            end
            break
        elseif answer == "q"
            break
        end
    end

    return nothing
end

get_calibration("Data/Raw/blade.h5", x, y, w, max_modulation, x_period, y_period, camera, slm)
##


##
confidence_inter = confint(fit; level=0.99)
##
img = capture(camera)
visualize(img)

#sum(img .- 2) / 64
##
x_derivative = Matrix{Int}(undef, size(img, 1) - 1, size(img, 2))
visualize(x_derivative)

for n ∈ axes(x_derivative, 1)
    for m ∈ axes(x_derivative, 2)
        x_derivative[n, m] = img[n+1, m] - img[n, m]
    end
end

vec_x_derivative = vec(sum(x_derivative, dims=2))
pos = argmax(vec_x_derivative)

fig = Figure(size=(1080, 1080), figure_padding=0)
ax = Axis(fig[1, 1], aspect=DataAspect())
hidedecorations!(ax)
heatmap!(ax, img, colormap=:jet)
vlines!(ax, pos + 1, color=:red, linewidth=4)
fig
##
x_b = (pos + 1 - fit.param[1]) / fit.param[3]
##
function model(x, y, pars)
    x₀, y₀, w, α, amplitude, offset, x_b = pars

    @. offset + amplitude * abs2(hg(x - x₀, α * (y' - y₀); w, include_normalization=false)) * (x < x_b)
end

visualize(model(x_cam, y_cam, vcat(fit.param, pos)))
##


function basis_loop(n_modes, n_masks, basis_functions,
    saving_path, saving_name, incoming,
    slm, camera, x, y, max_modulation, xperiod, yperiod; sleep_time=0.15)

    file = h5open(saving_path, "cw")

    basis = stack(f(x, y) for f ∈ basis_functions)
    ρs = file["labels_dim$(length(basis_functions))"][:, :, :]
    eigen_states = similar(basis)
    desireds = Array{ComplexF32,3}(undef, size(basis, 1), size(basis, 2), n_masks)

    width = pyconvert(Int, camera.camera.get_param("width"))
    height = pyconvert(Int, camera.camera.get_param("height"))
    output = Array{UInt8,3}(undef, width, height, n_masks)

    result = Array{UInt8,3}(undef, width, height, n_modes)

    for n ∈ axes(ρs, 3)
        @info "Processing mode $n"
        ρ = view(ρs, :, :, n)
        F = eigen(Hermitian(ρ))
        @tullio eigen_states[j, k, n] = F.vectors[m, n] * basis[j, k, m]

        generate_fields!(desireds, eigen_states, F.values)
        loop_capture!(output, desireds, incoming, slm, camera, x, y, max_modulation, xperiod, yperiod; sleep_time)

        map!(x -> round(UInt8, x), view(result, :, :, n), mean(output, dims=3))
    end

    file["images_"*saving_name] = result
    #file["labels_"*saving_name] = ρs
    close(file)
end

basis_functions = positive_l_basis(2, [0, 0, w, 1])

basis_loop(100, 300, basis_functions,
    "Data/Raw/blade.h5", "small", incoming,
    slm, camera, x, y, 82, 5, 4; sleep_time=0.05)
##

