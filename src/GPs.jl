__precompile__(true)

"""
A general purpose Gaussian Process implementation for Julia.
"""
module GPs

using StaticArrays
using DocStringExtensions

include("utils.jl")
include("Metrics.jl")
include("Kernels.jl")
include("Covariance.jl")


#--- Gaussian Processes

"""
An abstract Gaussian Process type. Every child type should implement the
following functions.

    `?`
"""
abstract type AbstractGP end

export AbstractGP


end # module
