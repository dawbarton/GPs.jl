module Covariance

using ..Kernels

function covariance_matrix!(K::UpperTriangular, kernel::Kernel, x)
    for j in 1:length(x)
        for i in 1:j
            K[i, j] = covariance(kernel, x[i], x[j])
        end
    end
    return K
end

function covariance_matrix!(K::AbstractMatrix, kernel::Kernel, x)
    n = length(x)
    for j in 1:n
        for i in 1:j
            K[i, j] = covariance(kernel, x[i], x[j])
        end
    end
    for j in 1:n
        for i in j+1:n
            K[i, j] = K[j, i]
        end
    end
    return K
end

function covariance_matrix!(K::AbstractMatrix, kernel::Kernel, x1, x2)
    for j in 1:length(x2)
        for i in 1:length(x1)
            K[i, j] = covariance(kernel, x1[i], x2[j])
        end
    end
    return K
end

covariance_matrix(kernel::Kernel, x::AbstractVector{T}) where {T <: AbstractVector{S}} where {S <: Number} = covariance_matrix!(Matrix{S}(length(x), length(x)), kernel, x)
covariance_matrix(kernel::Kernel, x1::AbstractVector{T}, x2::AbstractVector{T}) where {T <: AbstractVector{S}} where {S <: Number} = covariance_matrix!(Matrix{S}(length(x1), length(x2)), kernel, x1, x2)

end
