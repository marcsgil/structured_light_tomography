using QuantumMeasurements, Random, LinearAlgebra, CUDA
Random.seed!(0)

includet("../Utils/basis.jl")
includet("../Utils/model_fitting.jl")

x = Float32.(1:224)
y = Float32.(1:224)
rs = Iterators.product(x, y)
param = (1.0f0, 112.0f0, 112.0f0, 20.0f0)

μ = empty_measurement(length(rs), 6, Matrix{Float32})
μ2 = empty_measurement(length(rs), 6, Matrix{Float32})
buffers = Matrix{ComplexF32}(undef, 6, 512)

freqs = rand(Float32, length(x) * length(y))

@info "Center of Mass and Waist:"
@benchmark center_of_mass_and_waist($freqs, 2)

@info "Measurement update:"
display(@benchmark multithreaded_update_measurement!($μ, $buffers, $rs, $param, $positive_l_basis!))

@info "Linear Inversion Estimation:"
mthd = LinearInversion()
display(@benchmark estimate_state($freqs, $μ, $mthd))

@info "Normal Equations Estimation:"
mthd = NormalEquations(μ)
display(@benchmark estimate_state($freqs, $μ, $mthd))

@info "Pre Allocated Linear Inversion Estimation:"
mthd = PreAllocatedLinearInversion(μ)
display(@benchmark estimate_state($freqs, $μ, $mthd))

@info "Maximum Likelihood Estimation:"
mthd = MaximumLikelihood()
freqs = normalize(simulate_outcomes(I(6) / 6.0f0, μ, 2048), 1)
display(@benchmark estimate_state($freqs, $μ, $mthd))

if CUDA.functional()
    cu_freqs = cu(freqs)
    cu_μ = cu(μ)

    @info "Linear Inversion Estimation (CUDA):"
    cu_mthd = LinearInversion()
    display(@benchmark CUDA.@sync estimate_state($cu_freqs, $cu_μ, $cu_mthd))

    @info "Normal Equations Estimation (CUDA):"
    cu_mthd = NormalEquations(cu_μ)
    display(@benchmark CUDA.@sync estimate_state($cu_freqs, $cu_μ, $cu_mthd))

    @info "Pre Allocated Linear Inversion Estimation (CUDA):"
    mthd = PreAllocatedLinearInversion(μ)
    cu_mthd = PreAllocatedLinearInversion((cu(getproperty(mthd, name)) for name ∈ propertynames(mthd))...)

    display(@benchmark CUDA.@sync estimate_state($cu_freqs, $cu_μ, $cu_mthd))
end