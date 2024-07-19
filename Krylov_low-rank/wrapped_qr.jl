
@inline @views function thin_qr!(A::Matrix{T}) where {T <: AbstractFloat}
    # The in-place QR decomposition returns a compact
    # type for Q, which needs to be cast as a matrix
    Q, R = qr!(A)
    Q = Matrix{T}(Q)
    return Q, R
end

# Conditionally define the function for CuMatrix if CUDA is available
@static if @isdefined(CUDA)
    @inline @views function thin_qr!(A::CuMatrix{T}) where {T <: AbstractFloat}
        Q, R = qr!(A)
        Q = CuMatrix{T}(Q)
        return Q, R
    end
end

# Conditionally define the function for CuMatrix if CUDA is available
@static if @isdefined(AMDGPU)
    @inline @views function thin_qr!(A::ROCMatrix{T}) where {T <: AbstractFloat}
        Q, R = qr!(A)
        Q = ROCMatrix{T}(Q)
        return Q, R
    end
end