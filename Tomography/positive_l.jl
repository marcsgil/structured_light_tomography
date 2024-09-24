using BayesianTomography, HDF5, ProgressMeter
includet("../Utils/basis.jl")
includet("../Utils/position_operators.jl")
includet("../Utils/model_fitting.jl")
includet("../Utils/bootstraping.jl")
##
function load_data(path, key, bg)
    images, ρs = h5open(path) do file
        obj = file[key]
        read(obj), attrs(obj)["density_matrices"]
    end

    remove_background!(images, bg)

    images, ρs
end

path = "Data/positive_l.h5"

calibration = h5open(path) do file
    file["calibration"] |> read
end

x = axes(calibration, 1)
y = axes(calibration, 2)

fit = calibration_fit(x, y, calibration)

bg = round(UInt8, fit.param[5])

fit.param
##
dims = 2:6

fid = Matrix{Float64}(undef, 100, length(dims))

@showprogress for (n, dim) ∈ enumerate(dims)
    images, ρs = load_data(path, "images_dim$dim", bg)

    basis = positive_l_basis(dim, fit.param)
    povm = assemble_position_operators(x, y, basis)
    problem = StateTomographyProblem(povm)
    mthd = LinearInversion(problem)

    Threads.@threads for m ∈ axes(images, 3)
        probs = @view images[:, :, m]
        ρ = @view ρs[:, :, m]
        pred_ρ = prediction(probs, mthd)[1]

        fid[m, n] = fidelity(ρ, pred_ρ)
    end
end

mean(fid, dims=1)
##
dims = 2:6

fid_no_calib = Matrix{Float64}(undef, 100, length(dims))

p = Progress(prod(size(fid_no_calib)))
Threads.@threads for n ∈ eachindex(dims)
    dim = dims[n]
    images, ρs = load_data(path, "images_dim$dim", 0x02)

    Threads.@threads for m ∈ axes(images, 3)
        probs = @view images[:, :, m]
        ρ = @view ρs[:, :, m]

        param = center_of_mass_and_waist(probs, 2 * (dim - 1))
        basis = positive_l_basis(dim, param)
        povm = assemble_position_operators(x, y, basis)
        problem = StateTomographyProblem(povm)
        mthd = LinearInversion(problem)

        pred_ρ = prediction(probs, mthd)[1]

        fid_no_calib[m, n] = fidelity(ρ, pred_ρ)
        next!(p)
    end
end
finish!(p)

mean(fid_no_calib, dims=1)
##
h5open("Results/positive_l.h5", "cw") do file
    file["dims"] = collect(dims)
    file["fid"] = stack(bootstrap(slice) for slice ∈ eachslice(fid, dims=2))
    file["fid_no_calib"] = stack(bootstrap(slice) for slice ∈ eachslice(fid_no_calib, dims=2))
end