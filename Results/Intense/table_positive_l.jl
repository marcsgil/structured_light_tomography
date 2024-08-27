using HDF5, PrettyTables
includet("../../Utils/metrics.jl")
##
dims = 2:6
metrics = Matrix{Float32}(undef, 100, length(dims))
errors = similar(metrics)


for m ∈ axes(metrics, 2)
    θs, covs = h5open("Results/Intense/positive_l.h5") do file
        read(file["thetas_dim$(dims[m])"]), read(file["covs_dim$(dims[m])"])
    end

    for n ∈ axes(metrics, 1)
        θ = @view θs[:, 1, n]
        pred_θ = @view θs[:, 2, n]
        cov = @view covs[:, :, n]

        metrics[n, m], errors[n, m] = fidelity_metric(θ, pred_θ, cov, 0.95)
    end
end

sigdigits(x) = -floor(Int, log10(x))
##
mean_metrics = mean(metrics, dims=1) |> vec
mean_errors = mean(errors, dims=1) |> vec
mean_errors = map(mean_errors) do x
    round(x, sigdigits=1)
end

digits = sigdigits.(mean_errors)
mean_metrics = [round(x; digits) for (x, digits) ∈ zip(mean_metrics, digits)]

data = hcat(dims, 100*mean_metrics, 100*mean_errors)

pretty_table(data; 
header = ["Dimension", "Mean Fidelity", "Mean Error"], 
backend = Val(:latex),
hlines = :all)