using HDF5, PrettyTables
includet("../../Utils/metrics.jl")
sigdigits(x) = -floor(Int, log10(x))
##
metrics = Matrix{Float32}(undef, 100, 3)
errors = similar(metrics)
radius = h5open("Results/Intense/iris.h5") do file
    read(file["radius"])
end


for m ∈ axes(metrics, 2)
    θs, covs = h5open("Results/Intense/iris.h5") do file
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
J = sortperm(radius) |> reverse

mean_metrics = mean(metrics, dims=1)[J]
mean_errors = std(metrics, dims=1)[J]
mean_errors = map(mean_errors) do x
    round(x, sigdigits=1)
end

digits = sigdigits.(mean_errors)
mean_metrics = [round(x; digits) for (x, digits) ∈ zip(mean_metrics, digits)]

data = hcat(round.(radius[J], sigdigits=2), 100*mean_metrics, 100*mean_errors)

pretty_table(data; 
header = ["Iris Radius (w)", "Mean Fidelity", "Mean Error"], 
backend = Val(:latex),
hlines = :all)