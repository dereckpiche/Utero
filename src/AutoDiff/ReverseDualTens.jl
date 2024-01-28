""" TODOS
- Use Sparse Arrays
- Dont use arrays where not needed, 
simply use the linker.
"""



"""
    Chain(Jyx, Jzy)
Apply the chain rule for two Jacobians.
"""
GetOrder(X::AbstractArray) = length(size(X))


function Chain(Jyx::AbstractArray, Jzy::AbstractArray, OrderY::Int64)
    # TODO: fix
    xshape = size(Jyx)[OrderY+1:end]
    yshape = size(Jyx)[1:OrderY]
    zshape = size(Jzy)[1:GetOrder(Jzy)-OrderY]
    J = zeros(zshape..., xshape...)
    xIters = map(k -> 1:k, xshape)
    yIters = map(k -> 1:k, yshape)
    zIters = map(k -> 1:k, zshape)
    for (zindices, xindices) in Base.product(Base.product(zIters...), Base.product(xIters...))
        chainsum = 0.0
        for yindices in Base.product(yIters...)
            chainsum += Jyx[yindices..., xindices...] * Jzy[zindices..., yindices...]
        end
        J[zindices..., xindices...] = chainsum
    end
    return J
end

function Chain(Jyx::AbstractArray, Jzy::Real, OrderY::Int64)
    return Jyx * Jzy
end
#Chain(Jyx::Any, Jzy::Nothing, OrderY::Int64) = nothing
#Chain(Jyx::Nothing, Jzy::Any, OrderY::Int64) = nothing

Chain(Jyx::Real, Jzy::AbstractArray, OrderY::Int64) = Jyx * Jzy


GetOrder(X::AbstractArray) = length(size(X))

"""
    ⬅Dual(::typeof(f), x1::AbstractArray, x2::AbstractArray, ...)
On the left, return the result of the operation. 
With z = f(x1, x2, ...), on the right, return the 
'Chainer" function: ``Ṫ(Z) -> Ṫ(X1), Ṫ(X2), ...``
"""
function ⬅Dual(::typeof(*), X::AbstractMatrix, Y::AbstractMatrix)
    # TODO: fix
    Z = X * Y
    (Mx, Nx) = size(X); (My, Ny) = size(Y)
    # Compute Tensobian Jacobian of Z with respect to X
    Jzx = zeros(Float64, Mx, Ny, Mx, Nx)
    for i in 1:Mx
        Jzx[i, 1:end, i, 1:end] = Y
    end
    # Compute Tensorial Jacobian of Z with respect to y
    Jzy = zeros(Float64, Mx, Ny, My, Ny)
    for i in 1:Ny
        Jzy[1:end, i, 1:end, i] = X
    end
    OrderZ = GetOrder(Z)
    return Z, Jcz -> (Chain(Jzx, Jcz, OrderZ), Chain(Jzy, Jcz, OrderZ))
end
@⬅BinaryFunctionOL Base.:*


function ⬅Dual(::typeof(+), X::AbstractArray, Y::AbstractArray)
    Z = X + Y
    shape = size(X)

    # Compute Tensobian Jacobian of Z with respect to X
    Jzx = zeros(Float64, shape..., shape...)
    for inds in Base.product(map(k -> 1:k, shape)...)
        Jzx[inds..., inds...] = Y[inds...]
    end

    # Compute Tensorial Jacobian of Z with respect to y
    Jzy = zeros(Float64, shape..., shape...)
    for inds in Base.product(map(k -> 1:k, shape)...)
        Jzy[inds..., inds...] = X[inds...]
    end

    OrderZ = GetOrder(Z)
    return Z, Jcz -> (Chain(Jzx, Jcz, OrderZ), Chain(Jzy, Jcz, OrderZ))
end
@⬅BinaryFunctionOL Base.:+


function ⬅Dual(::typeof(ReLU), X::AbstractArray)
    Z = ReLU(X)
    xshape = size(X)
    Jzx = zeros(Float64, size(X)..., size(X)...)
    for xinds in Base.product(map(k -> 1:k, xshape)...)
        X[xinds...] > 0 ? Jzx[xinds..., xinds...] = X[xinds...] : nothing
    end
    return Z, Jcz -> Chain(Jzx, Jcz, GetOrder(Z))
end
@⬅UnaryFunctionOL ReLU 

function ⬅Dual(::typeof(Sigmoid), X::AbstractArray)
    Z = ReLU(X)
    xshape = size(X)
    Jzx = zeros(Float64, size(X)..., size(X)...)
    for xinds in Base.product(map(k -> 1:k, xshape)...)
        Jzx[xinds..., xinds...] = Sigmoid(X[xinds...])*(1 - Sigmoid(X[xinds...]))
    end
    return Z, Jcz -> Chain(Jzx, Jcz, GetOrder(Z))
end
@⬅UnaryFunctionOL Sigmoid

function ⬅Dual(::typeof(map), f::Function, X::AbstractArray)
    # TODO
end




