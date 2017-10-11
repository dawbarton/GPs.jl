module GaussianLikelihoods

using ..GPs
using ..Kernels
using StaticArrays
using DocStringExtensions

import ..GPs: sethyperparameters!, gethyperparameters, hyperparametercount,
    optimizehyperparameters!, loglikelihood

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
    covariance_matrix!(gp.K, gp.kernel, gp.x)
    # Add measurement noise at each point
    n = length(gp.x)
    for i in 1:n+1:n*n
        gp.L[i] += gp.σn^2
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

function loglikelihood(gp::GaussianLikelihood)
    # log likelihood of the GP itself (w.r.t. the hyperparameters)
    n = length(gp.x)
    # log(det(K))
    logdetK = sum(log.(diag(gp.L)))
    # The log likelihood
    lik = 0.5mean(sum(gp.y.*gp.α, 1)) + logdetK + 0.5n*log(2π)
end

function loglikelihood_dθ(gp::GaussianLikelihood{T}) where {T}
    dθ = zeros(T, hyperparametercount(gp))
    n = length(gp.x)
    I = eye(T, n)
    At_ldiv_B!(gp.L, I) # I = gp.L' \ eye(n)
    A_ldiv_B!(gp.L, I) # I = gp.L \ (gp.L' \ eye(n))
    #inner = (gp.α*gp.α' - gp.L \ (gp.L' \ eye(n)))'
    for j in 1:n
        for i in 1:j
            K_dθ = covariance_dθ(gp.kernel, gp.x[i], gp.x[j], gp.K[i, j])
            dθ .+= K_dθ .* (mean(gp.α[i, :] .* gp.α[j, :]) - I[i, j]) * ifelse(i == j, 0.5*one(T), one(T)) # use ifelse to avoid computing the symmetric counterparts
        end
    end
    return dθ
end

#     inner = (alpha*alpha' - R \ (R' \ eye(n)))';
#     logL_theta = zeros(size(par));
#     logL_theta(1) = 0.5*sum(sum(inner.*K_sigma_n));
#     logL_theta(2) = 0.5*sum(sum(inner.*K_sigma_f));
#     l = par(3:end);
#     if length(l) == 1
#         logL_theta(3) = 0.5*sum(sum(inner.*K_l, 2));
#     else
#         for i = 1:length(l)
#             logL_theta(i + 2) = 0.5*sum(sum(inner.*K_l(:, :, i)));
#         end
#     end

end
