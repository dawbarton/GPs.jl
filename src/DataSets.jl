module DataSets

using ..GPs
using StaticArrays

"""
TrainingSet{NX, NY, T}

A data structure to store inputs (x) and outputs (y) for use in training a
Gaussian process. Wherever possible data is not copied but data structures are
reinterpreted for speed. However, this could lead to inadvertant data
modification by the user if sufficient care is not taken.
"""
struct TrainingSet{NX, NY, T}
    x::Vector{SVector{NX, T}}
    y::Vector{SVector{NY, T}}
    xm::Matrix{T}
    ym::Matrix{T}

    function TrainingSet(x::Vector{SVector{NX, T}}, y::Vector{SVector{NY, T}}) where {NX, NY, T <: Number}
        if length(x) != length(y)
            throw(ArgumentError("x and y must have the same length"))
        end
        new{NX, NY, T}(x,
                       y,
                       reinterpret(T, x, (NX, length(x))),
                       reinterpret(T, y, (NY, length(y))))
    end
    function TrainingSet(x::Matrix{T}, y::Matrix{T}) where {T <: Number}
        if size(x, 2) != size(y, 2)
            throw(ArgumentError("x and y must have the same number of columns"))
        end
        new{size(x, 1), size(y, 1), T}(reinterpret(MVector{size(x, 1), T}, x, (size(x, 2),)),
                                       reinterpret(MVector{size(y, 1), T}, y, (size(y, 2),)),
                                       x,
                                       y)
    end
end

TrainingSet(x, y) = TrainingSet(toSVector(x), toSVector(y))

export TrainingSet

samplecount(data::TrainingSet) = length(data.x)

export samplecount

end
