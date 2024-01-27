 ⬅DualedTens = []

"""
    Chain(Jxy, Jyz)
Apply the chain rule for two Jacobians.
"""
GetOrder(X::AbstractArray) = length(size(X))


function Chain(Jxy::AbstractArray, Jyz::AbstractArray, OrderY::Int64)
    xshape = size(Jxy)[OrderY+1,end]
    yshape = size(Jxy)[1, OrderY]
    zshape = size(Jyz)[1, GetOrder(Jyz)-OrderY]
    J = zeros(zshape, xshape)
    xIters = map(k -> 1:k, xshape)
    yIters = map(k -> 1:k, yshape)
    zIters = map(k -> 1:k, zshape)
    for (zindices, xindices) in zip(zip(zIters...), zip(xIters...))
        chainsum = 0.0
        for yindices in zip(yIters...)
            chainsum += Jxy[yindices..., xindices...] * Jyz[zindices..., yindices...]
        end
        J[zindices..., xindices...] = chainsum
    end
end

GetOrder(X::AbstractArray) = length(size(X))

"""
    ⬅Dual(::typeof(f), x1::AbstractArray, x2::AbstractArray, ...)
On the left, return the result of the operation. 
With z = f(x1, x2, ...), on the right, return the 
'Chainer" function: ``Ṫ(Z) -> Ṫ(X1), Ṫ(X2), ...``
"""
function ⬅Dual(::typeof(*), X::AbstractMatrix, Y::AbstractMatrix)
    Z = X * Y
    (Mx, Nx) = size(X); (My, Ny) = size(Y)
    # Compute Tensobian Jacobian of Z with respect to X
    Jxz = spzeros(Float64, Mx, Ny, Mx, Nx)
    for i in 1:Mx
        Jxz[i, 1:end, i, 1:end] = Y
    end
    # Compute Tensorial Jacobian of Z with respect to y
    Jyz = spzeros(Float64, Mx, Ny, My, Ny)
    for i in 1:Ny
        Jyz[1:end, i, 1:end, i] = X
    end
    OrderZ = GetOrder(Z)
    return z, Jzc -> (Chain(Jxz, Jzc, OrderZ), Chain(Jyz, Jzc, OrderZ))
end
@⬅OloadBinaryF Base.:*


function ⬅Dual(::typeof(+), X::AbstractArray, Y::AbstractArray)
    Z = X + Y
    shape = size(X)

    # Compute Tensobian Jacobian of Z with respect to X
    Jxz = spzeros(Float64, shape..., shape...)
    for inds in zip(map(k -> 1:k, shape)...)
        Jxz[inds..., inds...] = Y[inds]
    end

    # Compute Tensorial Jacobian of Z with respect to y
    Jyz = spzeros(Float64, shape..., shape...)
    for inds in zip(map(k -> 1:k, shape)...)
        Jyz[inds..., inds...] = X[inds]
    end

    OrderZ = GetOrder(Z)
    return z, Jzc -> (Chain(Jxz, Jzc, OrderZ), Chain(Jyz, Jzc, OrderZ))
end
@⬅OloadBinaryF Base.:+


function ⬅Dual(::typeof(ReLU), X::AbstractArray)
    Z = ReLU(X)
    xshape = size(X)
    Jxz = spzeros(Float64, size(X)..., size(X)...)
    for xinds in zip(map(k -> 1:k, xshape)...)
        X[xinds] > 0 ? Jx[xinds..., xinds...] = X[xinds] : nothing
    end
    return Z, Jzc -> Chain(Jxz, Jzc, GetOrder(Z))
end
@⬅OloadUnaryF ReLU 

function ⬅Dual(::typeof(map), f::Function, X::AbstractArray)
    # TODO
end




