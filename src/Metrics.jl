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

mutable struct Euclidean{Nx, T, L <: Union{T, SVector{Nx, T}}} <: Metric{T}
    ℓ::L
end

export Euclidean

sqrdistance(m::Euclidean, x1, x2) = sum(((x1 .- x2)./m.ℓ).^2)


end
