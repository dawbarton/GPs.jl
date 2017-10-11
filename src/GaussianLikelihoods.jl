module GaussianLikelihoods

using ..GPs
using ..Kernels
using StaticArrays
using DocStringExtensions

import ..GPs: sethyperparameters!, gethyperparameters, hyperparametercount

mutable struct GaussianLikelihood{T, NX, K <: Kernel{T}} <: AbstractGP{T}
    kernel::K # covariance function
    σn::T # additive noise (MVector so it remains mutable)
    x::Vector{SVector{NX, T}} # input data
    y::Matrix{T} # output data (rows are individual samples)
    L::UpperTriangular{T, Matrix{T}} # Cholesky factors of the covariance matrix
    α::Matrix{T} # α = K⁻¹y

    function GaussianLikelihood(kernel::Kernel, σn::T, x::Vector{SVector{NX, T}}, y::Matrix{T}) where {NX, T <: Number}
        n = length(x) # number of samples
        m = size(y, 2) # number of output dimension
        if n != size(y, 1)
            throw(DimensionMismatch("The number of samples in x and y must match (i.e. length(x) == size(y, 1))"))
        end
        gp = new{typeof(k), NX, T}(kernel,
                                   σn,
                                   Vector{SVector{NX, T}}(n), # x
                                   Matrix{T}(n, m), # y
                                   UpperTriangular(Matrix{T}(n, n)), # L = chol(K)
                                   Matrix{T}(n, m)) # α = K⁻¹y
        # Copy data to avoid unexpected problems in bad user code later
        gp.x .= x
        gp.y .= y' # assume we given samples as columns
        # Update with the specified hyperparameters/observations
        update!(gp)
        return gp
    end
end

GaussianLikelihood(kernel::Kernel{T}, σn::T, x, y) where {T} = GaussianLikelihood(kernel, σn, toSVector(x), toMatrix(y))

const GP = GaussianLikelihood

export GaussianLikelihood, GP

function sethyperparameters!(gp::GaussianLikelihood, θ)
    # Recursively set the hyperparameters
    gp.σn = θ[1]
    sethyperparameters!(gp.kernel, θ[2:end])
    # Update the GP with the new hyperparameters
    update!(gp)
end

gethyperparameters(gp::GaussianLikelihood) = [gp.σn[1]; gethyperparameters(gp.kernel)]

hyperparametercount(gp::GaussianLikelihood) = hyperparametercount(gp.kernel) + 1

function update!(gp::GP)
    # Prior covariance using the specified kernel
    covariance!(gp.L, gp.k, gp.x)
    # Add measurement noise at each point
    n = length(gp.x)
    for i in 1:n+1:n*n
        gp.L[i] += gp.σn^2
    end
    # Calculate the Cholesky decomposition
    cholfact!(Symmetric(gp.L)) # assumes L is upper triangular
    # Calculate quantities required for interpolation
    gp.α .= gp.y
    A_ldiv_B!(gp.L', gp.α) # L' \ y (intermediate)
    A_ldiv_B!(gp.L, gp.α) # α = L \ (L' \ y)
    return gp
end

#--- Posterior measures

import Base: mean, var, cov

"""
$(SIGNATURES)

Return the mean of the Gaussian Process at the specified points.
"""
function mean(gp::GaussianLikelihood{T}, x::Vector{SVector{N, T}}) where {T, N}
    Kₛ = covariance_matrix(gp.kernel, x, gp.x)
    return Kₛ*gp.α
end

mean(gp::GaussianLikelihood{T}, x::Union{Vector{T}, Matrix{T}}) where {T} = mean(gp, toSVector(x))

"""
$(SIGNATURES)

Return the variance of the Gaussian Process at the specified points.
"""
function var(gp::GaussianLikelihood{T}, x::Vector{SVector{N, T}}) where {T, N}
    Kₛ = covariance_matrix(gp.kernel, gp.x, x)
    v = gp.L' \ Kₛ
    Kₛₛ = [covariance(gp.kernel, xᵢ, xᵢ) for xᵢ in x]
    return Kₛₛ - vec(sum(v.*v, 1)) + gp.σn^2
end

var(gp::GaussianLikelihood{T}, x::Union{Vector{T}, Matrix{T}}) where {T} = var(gp, toSVector(x))

"""
$(SIGNATURES)

Return the covariance matrix of the Gaussian Process at the specified points.
"""
function cov(gp::GaussianLikelihood{T}, x::Vector{SVector{N, T}}) where {T, N}
    Kₛ = covariance_matrix(gp.kernel, gp.x, x)
    v = gp.L' \ Kₛ
    Kₛₛ = covariance_metrix(gp.kernel, x)
    Kₚ = Kₛₛ - v'*v
    for i in diagind(Kₚ)
        Kₚ[i] += gp.σn^2
    end
    return Kₚ
end

cov(gp::GaussianLikelihood{T}, x::Union{Vector{T}, Matrix{T}}) where {T} = cov(gp, toSVector(x))

end
