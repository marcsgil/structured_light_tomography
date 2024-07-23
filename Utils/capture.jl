function loop_capture!(output, desireds, incoming, slm, camera,
    x, y, max_modulation, xperiod, yperiod; sleep_time=0.15)
    N = size(desireds, 3)
    holo = generate_hologram(view(desireds, :, :, 1), incoming, x, y, max_modulation, xperiod, yperiod)
    @showprogress for n ∈ 1:N-1
        update_hologram(slm, holo; sleep_time)
        holo = generate_hologram(view(desireds, :, :, n + 1), incoming, x, y, max_modulation, xperiod, yperiod)
        capture!(view(output, :, :, n), camera)
    end

    update_hologram(slm, holo)
    capture!(view(output, :, :, N), camera)
    nothing
end

function basis_loop(basis_functions, ρs, n_masks,
    saving_path, saving_name, incoming,
    slm, camera, x, y, max_modulation, xperiod, yperiod; sleep_time=0.15, par=nothing)

    basis = stack(f(x, y) for f ∈ basis_functions)
    eigen_states = similar(basis)
    desireds = Array{ComplexF32,3}(undef, size(basis, 1), size(basis, 2), n_masks)

    width = pyconvert(Int, get_param(camera, "width"))
    height = pyconvert(Int, get_param(camera, "height"))
    output = Array{UInt8,3}(undef, width, height, n_masks)

    result = Array{UInt8,3}(undef, width, height, size(ρs, 3))

    for n ∈ axes(ρs, 3)
        @info "Processing mode $n"
        ρ = view(ρs, :, :, n)
        F = eigen(Hermitian(ρ))
        @tullio eigen_states[j, k, n] = F.vectors[m, n] * basis[j, k, m]

        generate_fields!(desireds, eigen_states, F.values)
        loop_capture!(output, desireds, incoming, slm, camera, x, y, max_modulation, xperiod, yperiod; sleep_time)

        map!(x -> round(UInt8, x), view(result, :, :, n), mean(output, dims=3))
    end

    h5open(saving_path, "cw") do file
        file[saving_name] = result
        obj = file[saving_name]

        attrs(obj)["density_matrices"] = ρs

        if !isnothing(par)
            attrs(obj)["par"] = par
        end
    end
end

function prompt_user(f, args...; prompt="Do you accept?", kwargs...)
    println(prompt)
    answer = readline()
    if answer == "y"
        f(args...; kwargs...)
    end
    answer ∈ ("y", "q")
end

function calculate_calibration_fit(calibration, saving_path, object_name="fit_param")
    x = axes(calibration, 1)
    y = axes(calibration, 2)

    p0 = Float64.([length(x) ÷ 2, length(y) ÷ 2, length(x) ÷ 10, 1, maximum(calibration), minimum(calibration)])

    fit = surface_fit(gaussian_model, x, y, calibration, p0)

    h5open(saving_path, "cw") do file
        file[object_name] = fit.param
        attrs(file[object_name])["x"] = collect(x)
        attrs(file[object_name])["y"] = collect(y)
    end

    nothing
end

function display_calibration(w, incoming, x, y, max_modulation, x_period, y_period, slm)
    desired = hg(x, y; w)
    holo = generate_hologram(desired, incoming, x, y, max_modulation, x_period, y_period)
    update_hologram(slm, holo)
end

function get_calibration(saving_path, incoming, x, y, w, max_modulation, x_period, y_period, camera, slm)
    display_calibration(w, incoming, x, y, max_modulation, x_period, y_period, slm)

    should_quit = false
    while !should_quit
        calibration = capture(camera)
        display(visualize(calibration))
        should_quit = prompt_user(calculate_calibration_fit, calibration, saving_path)
    end

    return nothing
end

function finite_diff_x_derivative(img)
    x_derivative = Matrix{Int}(undef, size(img, 1) - 1, size(img, 2))

    for n ∈ axes(x_derivative, 1)
        for m ∈ axes(x_derivative, 2)
            x_derivative[n, m] = Int(img[n+1, m]) - Int(img[n, m])
        end
    end

    x_derivative
end

function finite_diff_derivative(img)
    derivative = Vector{Int}(undef, length(img) - 1)

    for n ∈ eachindex(derivative)
        derivative[n] = Int(img[n+1]) - Int(img[n])
    end

    derivative
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

function normalized_position(pos, x₀, w)
    (pos - x₀) / w
end

function length_previous_modes(path, fragment)
    h5open(path) do file
        filter(x -> occursin(fragment, x), keys(file)) |> length
    end
end

function test_length_previous_modes()
    path = tempname() * ".h5"
    fragment = "test"

    for n ∈ 1:10
        h5open(path, "cw") do file
            file["$fragment$n"] = rand(10, 10)
        end
        @show length_previous_modes(path, fragment)
        @assert length_previous_modes(path, fragment) == n
    end
end


function prompt_blade_measurement(saving_path, density_matrix_path, n_masks,
    incoming, x, y, w, max_modulation, x_period, y_period, camera, slm; sleep_time=0.15)

    display_calibration(w, incoming, x, y, max_modulation, x_period, y_period, slm)

    fit_param = h5open(saving_path) do file
        if "fit_param" ∈ keys(file)
            file["fit_param"] |> read
        else
            error("No calibration data found in the file.")
        end
    end

    should_quit = false
    while !should_quit
        img = capture(camera)
        blade_pos = find_blade_position(img)
        display_img_and_vline(img, blade_pos)
        normalized_blade_pos = normalized_position(blade_pos, fit_param[1], fit_param[3])
        println("Normalized blade position: $normalized_blade_pos")

        basis_functions = positive_l_basis(2, [0, 0, w, 1])
        ρs = h5open(density_matrix_path) do file
            file["labels_dim$(length(basis_functions))"] |> read
        end

        fragment = "images_"
        previous_mode = length_previous_modes(saving_path, fragment)

        saving_name = fragment * "$(previous_mode + 1)"

        should_quit = prompt_user(basis_loop,
            basis_functions, ρs, n_masks,
            saving_path, saving_name, incoming,
            slm, camera, x, y, max_modulation, x_period, y_period; sleep_time, par=[blade_pos, normalized_blade_pos])
    end
end

function find_iris_bounding_square(img, x₀, y₀)
    x_derivative = finite_diff_derivative(view(img, :, round(Int, y₀)))
    y_derivative = finite_diff_derivative(view(img, round(Int, x₀), :))

    f(x, cutoff) = [abs(x[n]) * (n < cutoff) for n ∈ eachindex(x)]
    g(x, cutoff) = [abs(x[n]) * (n > cutoff) for n ∈ eachindex(x)]

    x₋ = f(x_derivative, x₀) |> argmax
    x₊ = g(x_derivative, x₀) |> argmax
    y₋ = f(y_derivative, y₀) |> argmax
    y₊ = g(y_derivative, y₀) |> argmax

    x₋, x₊, y₋, y₊
end

function find_iris_radius(img, x₀, y₀)
    x₋, x₊, y₋, y₊ = find_iris_bounding_square(img, x₀, y₀)
    (x₊ - x₋ + y₊ - y₋) / 4
end

function display_img_and_circle(img, radius, x₀, y₀)
    fig = Figure(size=(1080, 1080), figure_padding=0)
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hidedecorations!(ax)
    heatmap!(ax, img, colormap=:jet)
    θs = LinRange(0, 2π, 512)
    xs = x₀ .+ radius * cos.(θs)
    ys = y₀ .+ radius * sin.(θs)
    lines!(ax, xs, ys, color=:red, linewidth=4)
    display(fig)
end

function display_img_and_square(img, x₋, x₊, y₋, y₊)
    fig = Figure(size=(1080, 1080), figure_padding=0)
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hidedecorations!(ax)
    heatmap!(ax, img, colormap=:jet)
    vlines!(ax, [x₋, x₊], color=:red, linewidth=4)
    hlines!(ax, [y₋, y₊], color=:red, linewidth=4)
    display(fig)
end

function display_img_and_square(img, x₊, y₊)
    fig = Figure(size=(1080, 1080), figure_padding=0)
    ax = Axis(fig[1, 1], aspect=DataAspect())
    hidedecorations!(ax)
    heatmap!(ax, img, colormap=:jet)
    vlines!(ax, x₊, color=:red, linewidth=4)
    hlines!(ax, y₊, color=:red, linewidth=4)
    display(fig)
end


function prompt_iris_measurement(saving_path, density_matrix_path, n_masks,
    incoming, x, y, w, max_modulation, x_period, y_period, camera, slm; sleep_time=0.15)

    display_calibration(w, incoming, x, y, max_modulation, x_period, y_period, slm)

    fit_param = h5open(saving_path) do file
        if "fit_param" ∈ keys(file)
            file["fit_param"] |> read
        else
            error("No calibration data found in the file.")
        end
    end

    should_quit = false
    while !should_quit
        img = capture(camera)
        radius = find_iris_radius(img, fit_param[1], fit_param[2])
        display_img_and_circle(img, radius, fit_param[1], fit_param[2])
        #x₋, x₊, y₋, y₊ = find_iris_bounding_square(img, fit_param[1], fit_param[2])
        #display_img_and_square(img, x₋, x₊, y₋, y₊)
        normalized_radius = radius / fit_param[3]
        println("Normalized radius: $normalized_radius")

        basis_functions = positive_l_basis(2, [0, 0, w, 1])
        ρs = h5open(density_matrix_path) do file
            file["labels_dim$(length(basis_functions))"] |> read
        end

        fragment = "images_"
        previous_mode = length_previous_modes(saving_path, fragment)

        saving_name = fragment * "$(previous_mode + 1)"

        should_quit = prompt_user(() -> nothing)

        """should_quit = prompt_user((basis_loop),
            basis_functions, ρs, n_masks,
            saving_path, saving_name, incoming,
            slm, camera, x, y, max_modulation, x_period, y_period; sleep_time, par=[blade_pos, normalized_blade_pos])"""
    end
end