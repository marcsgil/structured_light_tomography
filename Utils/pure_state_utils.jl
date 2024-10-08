function cat_real_and_imag(ψs)
    vcat(real.(ψs), imag.(ψs))
end

function decat_real_and_imag(y::AbstractVector)
    @. @views y[1:end÷2] + im * y[end÷2+1:end]
end

function decat_real_and_imag(y::AbstractMatrix)
    @. @views y[1:end÷2, :] + im * y[end÷2+1:end, :]
end