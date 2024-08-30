using HDF5, PrettyTables
includet("../../Utils/metrics.jl")
##
dims = 2
metrics = Vector{Float32}(undef, length(dims))
errors = similar(metrics)


pred_θs = h5open("Results/Intense/positive_l_ml.h5") do file
    read(file["thetas"])
end

θs = h5open("Results/Intense/positive_l.h5") do file
    file["thetas_dim2"][:, 1, :]
end

for n ∈ axes(metrics, 1)
    θ = @view pred_θs[:, n]
    pred_θ = @view θs[:, n]

    metrics[n], errors[n] = fidelity_metric(θ, pred_θ, I, 0.95)
end


mean_metrics = mean(metrics, dims=1) |> vec
