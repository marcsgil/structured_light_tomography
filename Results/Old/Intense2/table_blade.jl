using HDF5, PrettyTables
includet("../../Utils/metrics.jl")
sigdigits(x) = -floor(Int, log10(x))
##
metrics = Matrix{Float32}(undef, 100, 4)
errors = similar(metrics)
blade_pos = h5open("Results/Intense/blade.h5") do file
    read(file["blade_pos"])
end


for m ∈ axes(metrics, 2)
    θs, covs = h5open("Results/Intense/blade.h5") do file
        read(file["thetas_$m"]), read(file["covs_$m"])
    end

    for n ∈ axes(metrics, 1)
        θ = @view θs[:, 1, n]
        pred_θ = @view θs[:, 2, n]
        cov = @view covs[:, :, n]

        metrics[n, m], errors[n, m] = fidelity_metric(θ, pred_θ, cov, 0.95)
    end
end
##
J = sortperm(blade_pos) |> reverse

mean_metrics = mean(metrics, dims=1)[J]
mean_errors = std(metrics, dims=1)[J]
mean_errors = map(mean_errors) do x
    round(x, sigdigits=1)
end

digits = sigdigits.(mean_errors)
mean_metrics = [round(x; digits) for (x, digits) ∈ zip(mean_metrics, digits)]

data = hcat(round.(blade_pos[J], sigdigits=2), 100*mean_metrics, 100*mean_errors)

pretty_table(data; 
header = ["Blade Position (w)", "Mean Fidelity", "Mean Error"], 
backend = Val(:latex),
hlines = :all)