using PartiallyCoherentSources, KernelAbstractions, LinearAlgebra, ProgressMeter, LsqFit


includet("../../Utils/model_fitting.jl")

#Loop capture

function loop_capture!(img_buffer, desireds, camera, slm, config; sleep_time=0.15)
    s1 = eachslice(desireds, dims=3)
    s2 = eachslice(img_buffer, dims=3)

    incoming = config.incoming
    x = config.x
    y = config.y
    max_modulation = config.max_modulation
    x_period = config.x_period
    y_period = config.y_period
    dest = centralized_cut(slm.image, (size(desireds, 1), size(desireds, 2)))

    @showprogress for (desired, img) ∈ zip(s1, s2)
        generate_hologram!(dest, desired, incoming, x, y, max_modulation, x_period, y_period)
        update_hologram!(slm; sleep_time)
        capture!(img, camera)
    end
end

@kernel function eigen_states_kernel!(eigen_states, vectors, basis)
    j, k, n = @index(Global, NTuple)

    tmp = zero(eltype(eigen_states))

    for m ∈ axes(vectors, 1)
        tmp += vectors[m, n] * basis[j, k, m]
    end

    eigen_states[j, k, n] = tmp
end

function basis_loop!(mean_imgs, img_buffer, desireds, eigen_states, basis, ρs,
    camera, slm, config; sleep_time=0.15)
    backend = get_backend(basis)
    kernel! = eigen_states_kernel!(backend)

    s1 = eachslice(ρs, dims=3)
    s2 = eachslice(mean_imgs, dims=3)


    for (n, (ρ, mean_img)) ∈ enumerate(zip(s1, s2))
        @info "Capturing mode $n"
        F = eigen(Hermitian(ρ))
        kernel!(eigen_states, F.vectors, basis; ndrange=size(eigen_states))
        generate_fields!(desireds, eigen_states, F.values)
        loop_capture!(img_buffer, desireds, camera, slm, config; sleep_time)


        for k ∈ axes(mean_img, 2), j ∈ axes(mean_img, 1)
            mean_img[j, k] = round(UInt8, mean(view(img_buffer, j, k, :)))
        end
    end
end

function basis_loop(basis_functions, ρs, n_masks, camera, slm, config; sleep_time=0.15)
    basis = stack(f(config.x, config.y) for f ∈ basis_functions)
    eigen_states = similar(basis)

    desireds = Array{eltype(basis),3}(undef, size(basis, 1), size(basis, 2), n_masks)

    width = pyconvert(Int, get_param(camera, "width"))
    height = pyconvert(Int, get_param(camera, "height"))
    img_buffer = Array{UInt8,3}(undef, width, height, n_masks)

    mean_imgs = Array{UInt8,3}(undef, width, height, size(ρs, 3))

    basis_loop!(mean_imgs, img_buffer, desireds, eigen_states, basis, ρs,
        camera, slm, config; sleep_time)

    mean_imgs
end

function save_basis_loop(saving_path, saving_name, args...; par=nothing, kwargs...)
    h5open(saving_path, "cw") do file
        file[saving_name] = basis_loop(args...; kwargs...)
        obj = file[saving_name]

        attrs(obj)["density_matrices"] = args[2]

        if !isnothing(par)
            attrs(obj)["par"] = par
        end
    end
end

#Prompting

function prompt_user(f, args...; prompt="Do you accept?", kwargs...)
    println(prompt)
    answer = readline()
    if answer == "y"
        f(args...; kwargs...)
    end
    answer ∈ ("y", "q")
end

#Calibration

function save_calibration(calibration, saving_path)
    x = LinRange(-0.5, 0.5, size(calibration, 1))
    y = LinRange(-0.5, 0.5, size(calibration, 2))

    p0 = Float64.([0, 0, 0.1, 1, maximum(calibration), minimum(calibration)])

    fit = surface_fit(gaussian_model, x, y, calibration, p0)

    h5open(saving_path, "cw") do file
        file["calibration"] = calibration
        attrs(file["calibration"])["fit_param"] = fit.param
        attrs(file["calibration"])["x"] = collect(x)
        attrs(file["calibration"])["y"] = collect(y)
    end

    nothing
end

function display_calibration(w, slm, config)
    incoming = config.incoming
    x = config.x
    y = config.y
    max_modulation = config.max_modulation
    x_period = config.x_period
    y_period = config.y_period

    desired = hg(x, y; w)
    holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period)
    update_hologram!(slm, holo)
end

function prompt_calibration(saving_path, w, camera, slm, config)
    display_calibration(w, slm, config)

    should_quit = false
    while !should_quit
        calibration = capture(camera)
        display(visualize(calibration))
        should_quit = prompt_user(save_calibration, calibration, saving_path)
    end

    return nothing
end

#Blade

function finite_diff_x_derivative(img)
    x_derivative = Matrix{Int}(undef, size(img, 1) - 1, size(img, 2))

    for n ∈ axes(x_derivative, 1)
        for m ∈ axes(x_derivative, 2)
            x_derivative[n, m] = Int(img[n+1, m]) - Int(img[n, m])
        end
    end

    x_derivative
end

function find_blade_position(img)
    x_derivative = finite_diff_x_derivative(img)
    sum(abs, x_derivative, dims=2) |> vec |> argmax
end

function display_img_and_vline(img, vline_pos)
    fig = Figure(size=(1080, 1080), figure_padding=0)
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hidedecorations!(ax)
    heatmap!(ax, img, colormap=:jet)
    vlines!(ax, vline_pos, color=:red, linewidth=4)
    display(fig)
end

function length_previous_modes(path, fragment)
    h5open(path) do file
        filter(x -> occursin(fragment, x), keys(file)) |> length
    end
end

function prompt_blade_measurement(saving_path, ρs, n_masks, w, camera, slm, config; sleep_time=0.15)
    fit_param, x = h5open(saving_path) do file
        if "calibration" ∈ keys(file)
            obj = file["calibration"]
            attrs(obj)["fit_param"], attrs(obj)["x"]
        else
            error("No calibration data found in the file.")
        end
    end

    display_calibration(w, slm, config)

    should_quit = false
    while !should_quit
        img = capture(camera)
        blade_pos = find_blade_position(img)
        display_img_and_vline(img, blade_pos)
        normalized_blade_pos = (x[blade_pos] - fit_param[1]) / fit_param[3]
        println("Normalized blade position: $normalized_blade_pos")

        basis_functions = positive_l_basis(2, [0, 0, w, 1])

        fragment = "images_"
        previous_mode = length_previous_modes(saving_path, fragment)

        saving_name = fragment * "$(previous_mode + 1)"

        should_quit = prompt_user(save_basis_loop,
            saving_path, saving_name,
            basis_functions, ρs, n_masks, camera, slm, config;
            sleep_time, par=[blade_pos, normalized_blade_pos])
    end
end

#Iris

function finite_diff_r_derivative(img, x, y, x₀, y₀)
    r_derivative = Matrix{Float64}(undef, size(img, 1) - 1, size(img, 2) - 1)

    for n ∈ axes(r_derivative, 2)
        for m ∈ axes(r_derivative, 1)
            Δx = x[m] - x₀
            Δy = y[n] - y₀
            r = sqrt(Δx^2 + Δy^2)

            r_derivative[m, n] = (Δx * (Int(img[m+1, n]) - Int(img[m, n]))
                                  +
                                  Δy * (Int(img[m, n+1]) - Int(img[m, n]))) / r
        end
    end

    r_derivative
end

function find_iris_radius(img, x, y, x₀, y₀)
    r_derivative = finite_diff_r_derivative(img, x, y, x₀, y₀)
    idxs = argmin(r_derivative)
    sqrt((x[idxs[1]] - x₀)^2 + (y[idxs[2]] - y₀)^2)
end

function display_img_and_circle(img, radius, x, y, x₀, y₀)
    fig = Figure(size=(1080, 1080), figure_padding=0)
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hidedecorations!(ax)
    heatmap!(ax, x, y, img, colormap=:jet)
    θs = LinRange(0, 2π, 512)
    xs = x₀ .+ radius * cos.(θs)
    ys = y₀ .+ radius * sin.(θs)
    lines!(ax, xs, ys, color=:red, linewidth=4)
    display(fig)
end

function prompt_iris_measurement(saving_path, ρs, n_masks, w, camera, slm, config; sleep_time=0.15)
    incoming = config.incoming
    x = config.x
    y = config.y
    max_modulation = config.max_modulation
    x_period = config.x_period
    y_period = config.y_period

    l = 2
    p = 0

    desired = lg(x, y; w, p, l)
    holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period)
    update_hologram!(slm, holo)

    fit_param, x, y = h5open(saving_path) do file
        if "calibration" ∈ keys(file)
            obj = file["calibration"]
            attrs(obj)["fit_param"], attrs(obj)["x"], attrs(obj)["y"]
        else
            error("No calibration data found in the file.")
        end
    end

    x₀ = fit_param[1]
    y₀ = fit_param[2]

    img = capture(camera)

    should_quit = false
    while !should_quit
        img = capture(camera)
        radius = find_iris_radius(img, x, y, x₀, y₀)
        normalized_radius = radius / fit_param[3]
        display_img_and_circle(img, radius, x, y, x₀, y₀)
        println("Normalized iris radius: $normalized_radius")

        basis_functions = positive_l_basis(2, [0, 0, w, 1])

        fragment = "images_"
        previous_mode = length_previous_modes(saving_path, fragment)

        saving_name = fragment * "$(previous_mode + 1)"

        should_quit = prompt_user(save_basis_loop,
            saving_path, saving_name,
            basis_functions, ρs, n_masks, camera, slm, config;
            sleep_time, par=[radius, normalized_radius])

        if should_quit
            h5open(saving_path, "cw") do file
                attrs(file[saving_name])["iris_calibration"] = img
            end
        end
    end
end