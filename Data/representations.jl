using BayesianTomography, LinearAlgebra

struct PureState end

function real_representation(c, ::PureState)
    #Represents the coeficients c as a real array.
    #We stack the real and then the imaginary part.
    D = size(c, 1)
    result = Array{real(eltype(c))}(undef, 2D, size(c, 2))
    result[1:D, :] = real.(@view c[1:end, :])
    result[D+1:2D, :] = imag.(@view c[1:end, :])
    result
end

function complex_representation(y, ::PureState)
    @views y[1:end÷2, :] + im * y[end÷2+1:end, :]
end

struct MixedState end