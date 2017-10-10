module Kernels

using ..Metrics
using StaticArrays

import ..GPs: sethyperparameters!, gethyperparameters, hyperparametercount

"""
An abstract kernel type. Every child type should implement the following
functions.

    `covariance(kernel, x1, x2)`
    `covariance_dθ(kernel, x1, x2, value)`
    `covariance_dx1(kernel, x1, x2, value)`
    `covariance_dx2(kernel, x1, x2, value)`
    `sethyperparameters!(kernel, θ)`
    `gethyperparameters(kernel)`
    `hyperparametercount(kernel)`
"""
abstract type Kernel{T <: Number} end

export Kernel

"""
`covariance(kernel, x1, x2)` returns the corresponding covariance kernel value.
"""
function covariance end

export covariance

"""
`covariance_dθ(kernel, x1, x2, cov)` returns the derivative w.r.t. the
hyperparameters of the corresponding covariance kernel evaluated at `(x1, x2)`.
The parameter `cov` is the previously computed covariance value.
"""
function covariance_dθ end

export covariance_dθ

"""
`covariance_dx1(kernel, x1, x2, cov)` returns the derivative w.r.t. x1 of the
corresponding covariance kernel evaluated at `(x1, x2)`. The parameter `cov` is
the previously computed covariance value.
"""
function covariance_dx end

export covariance_dy

"""
`covariance_dx2(kernel, x1, x2, cov)` returns the derivative w.r.t. x2 of the
corresponding covariance kernel evaluated at `(x1, x2)`. The parameter `cov` is
the previously computed covariance value.
"""
function covariance_dy end

export covariance_dy

#--- Square exponential kernels

"""
A square exponential type kernel of the form

```math
K(x_1, x_2) = \sigma_f^2\exp(d(x_1, x_2)/2)
```
where ``d(x_1, x_2)`` is an appropriate squared distance metric.

# Fields

    `metric::M`: the metric to be used.
    `σf::T`: the noise parameter.

# Type parameters

    `M <: Metric`: the type of metric used (can be inferred from metric).
    `T <: Number`: the base number type used (can be inferred from σf).

"""
mutable struct SqrExponential{M <: Metric, T <: Number} <: Kernel{T}
    metric::M
    σf::T
end

export SqrExponential

function covariance(kernel::SqrExponential, x1, x2)
    s² = sqrdistance(kernel.metric, x1, x2)
    return kernel.σf^2*exp(-0.5*s²)
end

function covariance_dθ(kernel::SqrExponential, x1, x2, cov)
    # cov is the previously computed covariance
    ds²dθ = sqrdistance_dθ(kernel.metric, x1, x2)
    # Derivative w.r.t. metric hyperparameters (chain rule)
    dcov_dθ = -0.5*ds²dθ*cov
    # Derivative w.r.t. noise parameter (uses SVector for speed)
    dcov_dσf = SVector(2*cov/kernel.σf)
    # Return concatenated array
    return [dcov_dσf; dcov_dθ]
end

function covariance_dx1(kernel::SqrExponential, x1, x2, cov)
    # cov is the previously computed covariance
    ds²dx = sqrdistance_dx1(kernel.metric, x1, x2)
    # Derivative w.r.t. x1 (chain rule)
    return -0.5*ds²dx*cov
end

function covariance_dx2(kernel::SqrExponential, x1, x2, cov)
    # cov is the previously computed covariance
    ds²dy = sqrdistance_dx2(kernel.metric, x1, x2)
    # Derivative w.r.t. x2 (chain rule)
    return -0.5*ds²dy*cov
end

function sethyperparameters!(kernel::SqrExponential, θ)
    kernel.σf = θ[1]
    sethyperparameters!(kernel.metric, θ[2:end])
end

gethyperparameters(kernel::SqrExponential) = [SVector(kernel.σf); gethyperparameters(kernel.metric)]

hyperparametercount(kernel::SqrExponential) = hyperparametercount(kernel.metric) + 1

end
