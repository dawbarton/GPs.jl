module GaussianLikelihoods

using ..GPs
using ..Kernels
using StaticArrays
using DocStringExtensions
using Optim

import ..GPs: sethyperparameters!, gethyperparameters, hyperparametercount,
    optimizehyperparameters!, loglikelihood, loglikelihood_dθ, loglikelihood_dθ!

mutable struct GaussianLikelihood{T, NX, NY, K <: Kernel{T}} <: AbstractGP{T}
    kernel::K # covariance function
    σn::T # additive noise (MVector so it remains mutable)
    x::Vector{SVector{NX, T}} # input data
    y::Matrix{T} # output data (rows are individual samples)
    K::UpperTriangular{T, Matrix{T}} # Covariance matrix
    L::UpperTriangular{T, Matrix{T}} # Cholesky factors of the covariance matrix
    α::Matrix{T} # α = K⁻¹y

    function GaussianLikelihood(kernel::Kernel, σn::T, x::Vector{SVector{NX, T}}, y::Matrix{T}) where {NX, T <: Number}
        n = length(x) # number of samples
        m = size(y, 1) # number of output dimensions
        if n != size(y, 2)
            throw(DimensionMismatch("The number of samples in x and y must match (i.e. length(x) == size(y, 1))"))
        end
        gp = new{T, NX, m, typeof(kernel)}(kernel,
                                           σn,
                                           Vector{SVector{NX, T}}(n), # x
                                           Matrix{T}(n, m), # y
                                           UpperTriangular(Matrix{T}(n, n)), # K
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

GaussianLikelihood(kernel::Kernel{T}, σn::T, x, y) where {T} = GaussianLikelihood(kernel, σn, toSVector(x)::Vector{SVector{NX, T}} where {NX}, toMatrix(y)::Matrix{T})

const GP = GaussianLikelihood

export GaussianLikelihood, GP

function sethyperparameters!(gp::GaussianLikelihood, θ)
    # Recursively set the hyperparameters
    gp.σn = θ[1]
    sethyperparameters!(gp.kernel, θ[2:end])
    # Update the GP with the new hyperparameters
    update!(gp)
end

gethyperparameters(gp::GaussianLikelihood) = [SVector(gp.σn[1]); gethyperparameters(gp.kernel)]

hyperparametercount(gp::GaussianLikelihood) = hyperparametercount(gp.kernel) + 1

function update!(gp::GP)
    # Prior covariance using the specified kernel
    covariance_matrix!(gp.K, gp.kernel, gp.x)
    # Add measurement noise at each point
    n = length(gp.x)
    for i in 1:n+1:n*n
        gp.K[i] += gp.σn^2
    end
    # Calculate the Cholesky decomposition
    copy!(gp.L, gp.K)
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
    return At_mul_Bt(gp.α, Kₛ) # (Kₛ*α)ᵀ
end

mean(gp::GaussianLikelihood{T}, x::Union{AbstractVector{T}, Matrix{T}}) where {T} = mean(gp, toSVector(x))

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

var(gp::GaussianLikelihood{T}, x::Union{AbstractVector{T}, Matrix{T}}) where {T} = var(gp, toSVector(x))

"""
$(SIGNATURES)

Return the covariance matrix of the Gaussian Process at the specified points.
"""
function cov(gp::GaussianLikelihood{T}, x::Vector{SVector{N, T}}) where {T, N}
    Kₛ = covariance_matrix(gp.kernel, gp.x, x)
    v = gp.L' \ Kₛ
    Kₛₛ = covariance_matrix(gp.kernel, x)
    Kₚ = Kₛₛ - v'*v
    for i in diagind(Kₚ)
        Kₚ[i] += gp.σn^2
    end
    return Kₚ
end

cov(gp::GaussianLikelihood{T}, x::Union{AbstractVector{T}, Matrix{T}}) where {T} = cov(gp, toSVector(x))

#--- Likelihoods

function loglikelihood(gp::GaussianLikelihood{T}) where {T}
    # log likelihood of the GP itself (w.r.t. the hyperparameters)
    n = length(gp.x)
    # log(det(K))
    logdetK = sum(log.(diag(gp.L)))
    # The log likelihood
    lik = T(0.5)*mean(sum(gp.y.*gp.α, 1)) + logdetK + T(0.5)*n*log(T(2)*π)
end

function loglikelihood_dθ!(dθ::AbstractVector{T}, gp::GaussianLikelihood{T}) where {T}
    n = length(gp.x)
    I = eye(T, n)
    At_ldiv_B!(gp.L, I) # I = gp.L' \ eye(n)
    A_ldiv_B!(gp.L, I) # I = gp.L \ (gp.L' \ eye(n))
    #inner = (gp.α*gp.α' - gp.L \ (gp.L' \ eye(n)))'
    Kdθ = @view dθ[2:end]
    for j in 1:n
        for i in 1:j
            mult = (mean(gp.α[i, :] .* gp.α[j, :]) - I[i, j]) * ifelse(i == j, T(0.5), T(1)) # use ifelse to avoid computing the symmetric counterparts
            Kdθ .-= covariance_dθ(gp.kernel, gp.x[i], gp.x[j]) .* mult
            dθ[1] -= ifelse(i == j, T(2) * gp.σn * mult, zero(T))
        end
    end
    return dθ
end

loglikelihood_dθ(gp::GaussianLikelihood{T}) where {T} = loglikelihood_dθ!(zeros(T, hyperparametercount(gp)), gp)

#--- Optimization

function optimizehyperparameters!(gp, solver = LBFGS(), options = Optim.Options())
    # Define objective functions with exponential scalings on the
    # hyperparameters as that seems to work best
    f = ϕ -> (θ = exp.(ϕ) ; θ != gethyperparameters(gp) ? sethyperparameters!(gp, θ) : nothing ; loglikelihood(gp))
    g! = (dϕ, ϕ) -> (θ = exp.(ϕ) ; println("ϕ: $(ϕ), θ: $(θ)") ; θ != gethyperparameters(gp) ? sethyperparameters!(gp, θ) : nothing ; loglikelihood_dθ!(dϕ, gp) ; dϕ .*= θ)

    θ₀ = Vector(gethyperparameters(gp))
    res = optimize(f, g!, log.(θ₀), solver, options)
    if Optim.converged(res)
        ϕ = Optim.minimizer(res)
        θ₀ .= exp.(ϕ)
        θ₀ != gethyperparameters(gp) ? sethyperparameters!(gp, θ₀) : nothing
        return true
    else
        sethyperparameters!(gp, θ₀)
        return false
    end
end

end
