__precompile__(true)

"""
A general purpose Gaussian Process implementation for Julia.
"""
module GPs

include("general.jl")

#--- Other submodules

include("Metrics.jl")
include("Kernels.jl")
include("DataSets.jl")
include("GaussianLikelihoods.jl")

using .Metrics
using .Kernels
using .DataSets
using .GaussianLikelihoods

export GP, TrainingSet, SqrExponential, Euclidean

end # module
