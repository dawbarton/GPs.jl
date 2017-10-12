using StaticArrays
using DocStringExtensions

#--- Generic functions used by multiple types

"""
Set the hyperparameters of an object.
"""
function sethyperparameters!
end

"""
Return the hyperparameters of an object.
"""
function gethyperparameters
end

"""
Return the number of hyper parameters an object has.
"""
function hyperparametercount
end

function optimizehyperparameters!
end

"""
`loglikelihood(gp)`

Return the negative log likelihood for a GP. Used in optimizing the
hyperparameters.

`loglikelihood(gp, x, y)`

Return the negative log likelihood of an observation given a GP.

"""
function loglikelihood
end

"""
Return the derivative of the negative log likelihood with respect to the
hyperparameters.
"""
function loglikelihood_dθ
end

"""
Return the derivative of the negative log likelihood with respect to the
sample positions.
"""
function loglikelihood_dx
end

export sethyperparameters!, gethyperparameters, hyperparametercount,
    optimizehyperparameters!, loglikelihood, loglikelihood_dθ, loglikelihood_dx

#--- Data conversion routines

"""
$(SIGNATURES)

Convert the input to a vector of SVectors.
"""
toSVector(x::Vector{T}) where {T <: Number} = reinterpret(SVector{1, T}, x, (length(x),))
toSVector(x::Vector{SVector{N, T}}) where {N, T <: Number} = x
toSVector(x::Matrix{T}) where {T <: Number} = reinterpret(SVector{size(x, 1), T}, x, (size(x, 2),))
toSVector(x::AbstractVector{T}) where {T <: Number} = [SVector{1}(el) for el in x] # catch-all for generators, etc

export toSVector

"""
$(SIGNATURES)

Convert the input to a Matrix.
"""
toMatrix(x::Vector{T}) where {T <: Number} = reshape(x, (1, length(x)))
toMatrix(x::Vector{SVector{N, T}}) where {N, T <: Number} = reinterpret(T, x, (N, length(x)))
toMatrix(x::Matrix{T}) where {T <: Number} = x

export toMatrix
