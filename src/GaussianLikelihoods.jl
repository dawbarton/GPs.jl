module GaussianLikelihoods

using ..GPs
using ..Kernels
using ..DataSets
using StaticArrays
using DocStringExtensions

import ..GPs: sethyperparameters!, gethyperparameters, hyperparametercount,
    loglikelihood, loglikelihood_dθ, loglikelihood_dθ!

struct GP{T, NX, NY, K <: Kernel{T}} <: AbstractGP{T}
    kernel::K # covariance function
    σn::MVector{1, T} # additive noise (MVector so it remains mutable)
    training::TrainingSet{NX, NY, T}
    K::UpperTriangular{T, Matrix{T}} # Covariance matrix
    L::UpperTriangular{T, Matrix{T}} # Cholesky factors of the covariance matrix
    α::Matrix{T} # α = K⁻¹y

    function GP(kernel::Kernel, σn::T, training::TrainingSet{NX, NY}) where {T <: Number, NX, NY}
        n = samplecount(training) # number of samples
        gp = new{T, NX, NY, typeof(kernel)}(kernel,
                                            MVector(σn),
                                            training,
                                            UpperTriangular(Matrix{T}(n, n)), # K
                                            UpperTriangular(Matrix{T}(n, n)), # L = chol(K)
                                            Matrix{T}(n, NY)) # α = K⁻¹y
        # Update with the specified hyperparameters/observations
        update!(gp)
        return gp
    end
end

export GP

function sethyperparameters!(gp::GP, θ)
    # Recursively set the hyperparameters
    gp.σn[1] = θ[1]
    sethyperparameters!(gp.kernel, θ[2:end])
    # Update the GP with the new hyperparameters
    update!(gp)
end

gethyperparameters(gp::GP) = [SVector(gp.σn); gethyperparameters(gp.kernel)]

hyperparametercount(gp::GP) = hyperparametercount(gp.kernel) + 1

function update!(gp::GP)
    # Prior covariance using the specified kernel
    covariance_matrix!(gp.K, gp.kernel, gp.training.x)
    # Add measurement noise at each point
    for i ∈ diagind(gp.K)
        gp.K[i] += gp.σn[1]^2
    end
    # Calculate the Cholesky decomposition
    copy!(gp.L, gp.K)
    cholfact!(Symmetric(gp.L)) # assumes L is upper triangular
    # Calculate quantities required for interpolation
    gp.α .= gp.ym'
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
function mean(gp::GP{T}, x::Vector{SVector{N, T}}) where {T, N}
    Kₛ = covariance_matrix(gp.kernel, x, gp.training.x)
    return At_mul_Bt(gp.α, Kₛ) # (Kₛ*α)ᵀ
end

mean(gp::GaussianLikelihood{T}, x::Union{AbstractVector{T}, Matrix{T}}) where {T} = mean(gp, toSVector(x))

"""
$(SIGNATURES)

Return the variance of the Gaussian Process at the specified points.
"""
function var(gp::GP{T}, x::Vector{SVector{N, T}}) where {T, N}
    Kₛ = covariance_matrix(gp.kernel, gp.training.x, x)
    v = gp.L' \ Kₛ
    Kₛₛ = [covariance(gp.kernel, xᵢ, xᵢ) for xᵢ in x]
    return Kₛₛ - vec(sum(v.*v, 1)) + gp.σn[1]^2
end

var(gp::GaussianLikelihood{T}, x::Union{AbstractVector{T}, Matrix{T}}) where {T} = var(gp, toSVector(x))

"""
$(SIGNATURES)

Return the covariance matrix of the Gaussian Process at the specified points.
"""
function cov(gp::GP{T}, x::Vector{SVector{N, T}}) where {T, N}
    Kₛ = covariance_matrix(gp.kernel, gp.training.x, x)
    v = gp.L' \ Kₛ
    Kₛₛ = covariance_matrix(gp.kernel, x)
    Kₚ = Kₛₛ - v'*v
    for i ∈ diagind(Kₚ)
        Kₚ[i] += gp.σn^2
    end
    return Kₚ
end

cov(gp::GaussianLikelihood{T}, x::Union{AbstractVector{T}, Matrix{T}}) where {T} = cov(gp, toSVector(x))

#--- Likelihoods

function loglikelihood(gp::GP{T}) where {T}
    # log likelihood of the GP itself (w.r.t. the hyperparameters)
    n = length(gp.training.x)
    # log(det(K))
    logdetK = sum(log.(diag(gp.L)))
    # The log likelihood
    lik = T(0.5)*mean(sum(gp.y.*gp.α, 1)) + logdetK + T(0.5)*n*log(T(2)*π)
end

function loglikelihood_dθ!(dθ::AbstractVector{T}, gp::GP{T}) where {T}
    n = length(gp.x)
    I = eye(T, n)
    At_ldiv_B!(gp.L, I) # I = gp.L' \ eye(n)
    A_ldiv_B!(gp.L, I) # I = gp.L \ (gp.L' \ eye(n))
    #inner = (gp.α*gp.α' - gp.L \ (gp.L' \ eye(n)))'
    Kdθ = @view dθ[2:end]
    for j ∈ 1:n
        for i ∈ 1:j
            mult = (mean(gp.α[i, :] .* gp.α[j, :]) - I[i, j]) * ifelse(i == j, T(0.5), T(1)) # use ifelse to avoid computing the symmetric counterparts
            Kdθ .-= covariance_dθ(gp.kernel, gp.x[i], gp.x[j]) .* mult
            dθ[1] -= ifelse(i == j, T(2) * gp.σn * mult, zero(T))
        end
    end
    return dθ
end

loglikelihood_dθ(gp::GP{T}) where {T} = loglikelihood_dθ!(zeros(T, hyperparametercount(gp)), gp)


end
