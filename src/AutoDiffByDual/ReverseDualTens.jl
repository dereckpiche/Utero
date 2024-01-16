⬅DualedTens = []

"""
    Tensobian
Special tensorized version of the Jacobian to store first order partial derivatives.
For the Tensobian Ṫ of a function Z = f(U), where U and Z are tensors of arbitrary order,
Ṫ.∂Z∂U[i1, i2, i3, ..., in] is the partial derivative of Z[i1, i2, ..., is] with respect to
U[is+1, is+2, ..., in]. Ṫ.split stores s.
"""
struct Tensobian
    ∂Z∂U::AbstractArray
    split::Int64 # Index of input output separation.
    function Tensobian(∂Z∂U::AbstractArray, split::Int64)
        return new(∂Z∂U, split)
    end
end

"""
    TbChain(x, y)
Apply the chain rule to two Tensobians.
"""
TbChain(x::AbstractMatrix, y::AbstractMatrix) = x*y
TbChain(x::AbstractArray, y::Real) = x*y
TbChain(x::AbstractArray, y::Real) = x*y
TbChain(x::Real, y::AbstractArray) = x*y

function TbChain(x::Tensobian, y::Tensobian)
    # TODO
end

"""
    ⬅Dual(::typeof(f), x1::AbstractArray, x2::AbstractArray, ...)
On the left, return the result of the operation. 
With z = f(x1, x2, ...), on the right, return the 
'Chainer" function: ``Ṫ(Z) -> Ṫ(X1), Ṫ(X2), ...``
"""
function ⬅Dual(::typeof(+), x::AbstractArray, y::AbstractArray)
    # TODO
    z = x + y
    Ṫzx = x
    Ṫzy = x
    return z, Ṫz -> (TbChain(Ṫzx, Ṫz), TbChain(Ṫzy, Ṫz))
end
push!(⬅DualedTens, :+)

function ⬅Dual(::typeof(*), X::AbstractMatrix, Y::AbstractMatrix)
    Z = X * Y
    (Mx, Nx) = size(X); (My, Ny) = size(Y)

    # Compute Tensobian of Z with respect to X
    Ṫx = Tensobian(spzeros(Float64, Mx, Ny, Mx, Nx), 2)
    for i in 1:Mx
        Ṫx.∂Z∂U[i, 1:end, i, 1:end] = Y
    end

    Ṫy = Tensobian(spzeros(Float64, Mx, Ny, Mx, Nx), 2)
    

    # Compute Tensobian of Z with respect to y
    Ṫzy = spzeros(Float64, Mx, Ny, Mx, Nx)



    return z, Ṫz -> (TbChain(Ṫzx, Ṫz), TbChain(Ṫzy, Ṫz))
end
push!(⬅DualedTens, :+)


function ⬅Dual(::typeof(cos), x::AbstractArray, y::AbstractArray)
    z = cos(x)
    ∂z∂x = -sin(x)
    return z, ∂l∂z -> ∂l∂z*∂z∂x
end
push!(⬅DualedTens, :cos)

