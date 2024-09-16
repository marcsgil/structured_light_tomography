using BayesianTomography, PositionMeasurements, LinearAlgebra, HDF5, Images, UnPack, JLD2, VectorBeamTomography, Lux
using CairoMakie

outer(x) = x * x'

function predict_pars(model, parameters, states, mode_name)
    pars = Array{Float32}(undef, 4, 2, 6)

    file = h5open("ExperimentalData/data.h5")

    for (i, pol) in enumerate(["H", "V", "A", "D", "R", "L"])
        for (j, side) ∈ enumerate(["direct", "converted"])
            x = imresize(read(file["$mode_name/I$(pol)_$(side)"]), 64, 64)
            normalize!(x, Inf)
            x = reshape(x, 64, 64, 1, 1)

            y, _ = Lux.apply(model, x |> gpu_device(), parameters |> gpu_device(), states |> gpu_device())

            pars[:, j, i] = y |> cpu_device() |> vec
        end
    end

    pars
end

function get_povm(mode_name)
    polarization_povm = outer.([
        [1, 0],
        [0, 1],
        [1, 1] / √2,
        [1, -1] / √2,
        [1, -im] / √2,
        [1, im] / √2
    ]) / 3

    file = jldopen("runs/best_model.jld2")
    @unpack parameters, states = file
    close(file)

    model = build_model((64, 64, 1), 4)
    y = predict_pars(model, parameters, states, mode_name)
    position_povm = Array{Matrix{ComplexF32}}(undef, 64, 64, 2, 6)
    mode_converter = diagm([ComplexF32(cis(k * π / 2)) for k ∈ 0:1])
    rs = LinRange(-0.5f0, 0.5f0, 64)

    for k ∈ 1:6, j ∈ 1:2
        basis = transverse_basis(1, y[1, j, k], y[2, j, k], y[3, j, k], y[4, j, k]) |> reverse
        position_povm[:, :, j, k] = assemble_position_operators(rs, rs, basis) / 2
        if j == 2
            unitary_transform!(view(position_povm, :, :, j, k), mode_converter)
        end
    end

    povm = similar(position_povm)

    for k ∈ 1:6, j ∈ 1:2, i ∈ 1:64, l ∈ 1:64
        povm[i, l, j, k] = kron(position_povm[i, l, j, k], polarization_povm[k])
    end

    povm
end

function predict_state(mode_name)
    file = h5open("ExperimentalData/data.h5")

    L = 64
    x = Array{Float32}(undef, L, L, 2, 6)

    for (k, pol) in enumerate(["H", "V", "A", "D", "R", "L"])
        for (j, side) ∈ enumerate(["direct", "converted"])
            x[:, :, j, k] = imresize(read(file["$mode_name/I$(pol)_$(side)"]), 64, 64)
        end
    end
    close(file)

    for j ∈ 1:2:size(x, 4)
        normalize!(view(x, :, :, :, j:j+1), 1)
    end
    normalize!(x, 1)

    povm = get_povm(mode_name)
    mthd = LinearInversion(povm)
    prediction(x, mthd) |> project2pure
end
##
ψ_pred = predict_state("LG-")

fidelity([1, 0, 0, 1] / √2, ψ_pred)
fidelity(kron([1, -im], [1, im]) / 2, ψ_pred)
fidelity([1, 0, 0, 0], ψ_pred)
##
mode_names = ["Hh", "LG+", "LG-", "PHI+"]
ϕs = [kron([1, 0], [1, 0]),
    kron([1, im], [1, -im]) / 2,
    kron([1, -im], [1, +im]) / 2,
    [1, 0, 0, 1] / √2]

for (mode_name, ϕ) ∈ zip(mode_names, ϕs)
    ψ = predict_state(mode_name)
    fid = round(fidelity(ψ, ϕ) * 100, sigdigits=3)
    println("Mode: $mode_name; Fidelity: $(fid)%")
end
##
m = 1

povm = get_povm(mode_names[m])
ψ = predict_state(mode_names[m])

sim = reshape(simulate_outcomes(ψ, povm, 10^7, atol=1e-2), (64, 64, 2, 6))
fig = Figure(size=(1200, 400))
for k ∈ axes(sim, 4)
    for j ∈ axes(sim, 3)
        ax = CairoMakie.Axis(fig[j, k], aspect=1)
        hidedecorations!(ax)
        heatmap!(sim[:, :, j, k], colormap=:hot)
    end
end
fig
##
file = h5open("ExperimentalData/data.h5")

L = 64
x = Array{Float32}(undef, L, L, 2, 6)

for (k, pol) in enumerate(["H", "V", "A", "D", "R", "L"])
    for (j, side) ∈ enumerate(["direct", "converted"])
        x[:, :, j, k] = imresize(read(file["$(mode_names[m])/I$(pol)_$(side)"]), 64, 64)
    end
end
close(file)

fig_exp = Figure(size=(1200, 400))
for k ∈ axes(sim, 4)
    for j ∈ axes(sim, 3)
        ax = CairoMakie.Axis(fig_exp[j, k], aspect=1)
        hidedecorations!(ax)
        heatmap!(x[:, :, j, k], colormap=:hot)
    end
end
fig_exp
