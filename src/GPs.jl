__precompile__(true)

"""
A general purpose Gaussian Process implementation for Julia.
"""
module GPs

#--- Gaussian Processes

"""
An abstract Gaussian Process type. Every child type should implement the
following functions.

    `?`
"""
abstract type AbstractGP{T <: Number} end

export AbstractGP

#--- Other submodules

include("utils.jl")
include("Metrics.jl")
include("Kernels.jl")
include("GaussianLikelihoods.jl")

end # module
