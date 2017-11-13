module Metrics

using StaticArrays

import ..GPs: sethyperparameters!, gethyperparameters, hyperparametercount

"""
An abstract metric type for those kernels that support the notion of a metric,
e.g., squared exponential kernels. Every child type should implement the
following functions.

    `sqrdistance(metric, x, y)`
    `sethyperparameters!(metric, θ)`
    `gethyperparameters(metric)`
    `hyperparametercount(metric)`
"""
abstract type Metric{T <: Number} end

export Metric

"""
`sqrdistance(metric, x1, x2)` returns the squared distance between `(x1, x2)`
according to the specified metric.
"""
function sqrdistance end

export sqrdistance

"""
`sqrdistance_dθ(metric, x1, x2)` returns the derivative w.r.t. the
hyperparameters of the squared distance between `(x1, x2)` according to the
specified metric.
"""
function sqrdistance_dθ end

export sqrdistance_dθ

"""
`sqrdistance_dx1(metric, x1, x2)` returns the derivative w.r.t. x1 of the
squared distance between `(x1, x2)` according to the specified metric.
"""
function sqrdistance_dx1 end

export sqrdistance_dx1

"""
`sqrdistance_dx2(metric, x1, x2)` returns the derivative w.r.t. x2 of the
squared distance between `(x1, x2)` according to the specified metric.
"""
function sqrdistance_dx2 end

export sqrdistance_dx2

#--- Standard Euclidean metric

"""
Euclidean{N, T}

The standard Euclidean metric in `N` dimensions. (Note that `N=1` corresponds to
an isotropic metric when the input data is multi-dimensional.)
"""
struct Euclidean{N, T} <: Metric{T}
    ℓ::MVector{N, T}
end
Euclidean(ℓ::Union{SVector, Number}) = Euclidean(MVector(ℓ))

export Euclidean

sqrdistance(metric::Euclidean, x1, x2) = sum(((x1 .- x2)./metric.ℓ).^2)

function sqrdistance_dθ(metric::Euclidean{1, T} where {T}, x1, x2)
    return SVector(-2*sum((x1 .- x2).^2)/metric.ℓ[1]^3)
end

function sqrdistance_dθ(metric::Euclidean{N, T} where {N, T}, x1, x2)
    return -2*(x1 .- x2).^2./metric.ℓ.^3
end

function sqrdistance_dx1(metric::Euclidean, x1, x2)
    return 2*(x1 .- x2)./metric.ℓ.^2
end

function sqrdistance_dx2(metric::Euclidean, x1, x2)
    return -2*(x1 .- x2)./metric.ℓ.^2
end

sethyperparameters!(metric::Euclidean, θ) = (metric.ℓ .= θ)
gethyperparameters(metric::Euclidean) = metric.ℓ
hyperparametercount(metric::Euclidean) = length(metric.ℓ)

end
