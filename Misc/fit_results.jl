using HDF5, CairoMakie, Optim, NonlinearSolve

function proposal_zela(n, pars)
    F_max = pars[1]
    #η = pars[2]
    #n₀ = pars[2]
    N = pars[2]

    #F_max - η * exp(-(n - n₀) / N)
    #F_max - η * exp(-n / N)
    #F_max - exp(-(n - n₀) / N)
    F_max - exp(-n / N)
end

function proposal_gil(n, pars)
    F_max = pars[1]
    N = pars[2]
    β = pars[3]

    F_max - (N / n)^β
end

function fit(proposal, fids, photocounts, p0)
    loss(pars) = sum((fid - proposal(n, pars))^2 for (fid, n) in zip(fids, photocounts))
    optimize(loss, p0)
end

file = h5open("Results/Photocount/machine_learning.h5")
fids = file["fids"] |> read
photocounts = file["photocounts"] |> read
close(file)
##
#p0 = [1, 1, 32, 50.0]
#p0 = [1, 1, 50.0]
#p0 = [1, 32, 50.0]
p0 = [1, 50.0]
#p0 = [1, 50.0, 5]
results = [fit(proposal_zela, fids[:, order], photocounts, p0) for order ∈ 1:4]

results[4].minimizer[2]
##
continuous_photocounts = [2^k for k ∈ LinRange(6, 11, 256)]

my_theme = Theme(
    fontsize=28,
    markersize=28,
    linewidth=3)
theme = merge(my_theme, theme_latexfonts())

with_theme(theme) do
    fig = Figure(resolution=(800, 500))
    ax = Axis(fig[1, 1],
        xlabel="Photocounts",
        ylabel="Mean Fidelity",
        xticks=photocounts,
        xscale=log2,)
    #yticks=0.88:0.01:1)

    ylims!(ax, 0, 1)

    colors = [:red, :blue, :green, :black]

    for order ∈ 1:4
        fit_fid = [proposal_zela(pc, results[order].minimizer) for pc ∈ continuous_photocounts]
        scatter!(ax, photocounts, fids[:, order], color=colors[order])
        lines!(ax, continuous_photocounts, fit_fid, color=colors[order])
    end

    fig
end
##
#lines([results[order].minimizer[2] for order ∈ 1:4])

with_theme(theme) do
    fig = Figure(resolution=(800, 500))
    ax = Axis(fig[1, 1],
        xlabel="Order",
        ylabel=L"N",
        yscale=log10,)
    #xscale = log10)

    scatter!(ax, [results[order].minimizer[2] for order ∈ 1:4])

    fig
end
##
# Step 1: Import necessary packages
using GLM, DataFrames

X = 1:4;
Y = [(results[order].minimizer[2]) for order ∈ 1:4];

# Step 2: Create or load your data
df = DataFrame(X=X, Y=Y);

# Step 3: Fit the model
model = lm(@formula(Y ~ X), df);

print("Exponential")
# Step 4: Extract the results
println("Coefficients:")
coef(model) # to get the coefficients
stderror(model) # to get the residual sum of squares
stderror(model) ./ coef(model)


##
f(n, pars) = proposal_zela(n, pars) - 0.90

with_theme(theme) do
    fig = Figure(resolution=(800, 500))
    ax = Axis(fig[1, 1],
        xlabel="Order",
        ylabel=L"n_{92}",
        yscale=log10)

    for order ∈ 1:4
        p = results[order].minimizer
        n0 = 0.9
        prob = NonlinearProblem(f, n0, p)
        sol = solve(prob)
        scatter!(ax, order, sol.u, color=:black)
    end

    fig
end
##
