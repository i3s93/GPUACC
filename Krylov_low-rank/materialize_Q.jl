"""
Wrapped calls to the thin QR decomposition. These calls convert the return Q from
a compact type to a dense matrix of the appropriate type.
"""

function materialize_Q(Q_factor::T, backend::CPU_backend) where {T}
   return Matrix(Q_factor) 
end

@static if @isdefined(CUDA)
    function materialize_Q(Q_factor::T, backend::CUDA_backend) where {T}
        return CuMatrix(Q_factor) 
    end
    materialize_Q(Q_factor::T, backend::CUDA_UVM_backend) where {T} = materialize_Q(Q_factor, CUDA_backend()) 
end




# function thin_qr!(A::Matrix{T}) where {T <: AbstractFloat}
#     Q, R = qr!(A)
#     Q = Matrix{T}(Q)
#     return Q, R
# end

# function thin_qr(A::Matrix{T}) where {T <: AbstractFloat}
#     Q, R = qr(A)
#     Q = Matrix{T}(Q)
#     return Q, R
# end

# @static if @isdefined(CUDA)
#     function thin_qr!(A::CuMatrix{T}) where {T <: AbstractFloat}
#         Q, R = qr!(A)
#         Q = CuMatrix{T}(Q)
#         return Q, R
#     end

#     function thin_qr(A::CuMatrix{T}) where {T <: AbstractFloat}
#         Q, R = qr(A)
#         Q = CuMatrix{T}(Q)
#         return Q, R
#     end
# end

# # Conditionally define the function for CuMatrix if CUDA is available
# @static if @isdefined(AMDGPU)
#     function thin_qr!(A::ROCMatrix{T}) where {T <: AbstractFloat}
#         Q, R = qr!(A)
#         Q = ROCMatrix{T}(Q)
#         return Q, R
#     end

#     function thin_qr(A::ROCMatrix{T}) where {T <: AbstractFloat}
#         Q, R = qr(A)
#         Q = ROCMatrix{T}(Q)
#         return Q, R
#     end
# end
