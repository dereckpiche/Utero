"""
================================================================
    ReverseDual.jl

PURPOSE:
Compute the reverse dual of basic functions.




TODOs:
- Include sparse arrays
================================================================
"""



"""
    Chain(Jyx, Jzy)
Apply the chain rule for two Tensorial Jacobians of arbitrary order.
"""
function Chain(Jyx::AbstractArray, Jzy::AbstractArray, OrderY::Int64)
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

Chain(Jyx::AbstractArray, Jzy::Real, OrderY::Int64) = Jyx * Jzy
Chain(Jyx::Real, Jzy::AbstractArray, OrderY::Int64) = Jyx * Jzy



"""
    ⬅Dual(::typeof(f), x1::AbstractArray, x2::AbstractArray, ...)
On the left, return the result of the operation. 
With z = f(x1, x2, ...), on the right, return the 
'Chainer" function: ``Ṫ(Z) -> Ṫ(X1), Ṫ(X2), ...``
"""

# ================================
# Element-Wise
# ================================

# =================== Addition
function ⬅Dual(::typeof(+), x::Number, y::Number)
    z = x + y
    ∂z∂x = 1
    ∂z∂y = 1
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
@⬅BinaryScalarFunctionOL Base.:+

function ⬅Dual(::typeof(+), X, Y)
    Z = X + Y
    return Z, ∇z -> (∇z, ∇z)
end
@⬅BinaryFunctionOL Base.:+

# =================== Substraction 

function ⬅Dual(::typeof(-), x::Number, y::Number)
    z = x - y
    ∂z∂x = 1
    ∂z∂y = -1
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
@⬅BinaryScalarFunctionOL Base.:-

function ⬅Dual(::typeof(-), X, Y)
    Z = X - Y
    return Z, ∇z -> (∇z, -∇z)
end
@⬅BinaryFunctionOL Base.:-

# =================== Multiplication


function ⬅Dual(::typeof(*), x::Number, y::Number)
    z = x * y
    ∂z∂x = y
    ∂z∂y = x
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
@⬅BinaryScalarFunctionOL Base.:*


function ⬅Dual(::typeof(broadcasted), ::typeof(*), X, Y)
    Z = X .* Y
    return Z, ∇Z -> (∇Z .* Y, ∇Z .* X)
end
@⬅BinaryBroadcastedOL Base.:*

# =================== Divison

function ⬅Dual(::typeof(/), x::Number, y::Number)
    z = x / y
    ∂z∂x = y
    ∂z∂y = -x/y^2
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
@⬅BinaryScalarFunctionOL Base.:/

function ⬅Dual(::typeof(broadcasted), ::typeof(/), X, Y)
    Z = X ./ Y
    return Z, ∇Z -> (∇Z ./ Y, ∇Z .* X) 
end
@⬅BinaryBroadcastedOL Base.:/

# =================== Exponentiation

function ⬅Dual(::typeof(^), x::Number, y::Number)
    z = x^y
    ∂z∂x = y*x^(y-1)
    ∂z∂y = log(x) * x^y 
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
@⬅BinaryScalarFunctionOL Base.:^

function ⬅Dual(::typeof(broadcasted), ::typeof(exp), X)
    Z = @. exp(X)
    return Z, ∇Z -> ∇Z .* Z
end
@⬅UnaryBroadcastedOL Base.exp

function ⬅Dual(::typeof(broadcasted), ::typeof(^), X, Y)
    Z = @. X ^ Y
    ∇X = @. Y * X^(Y-1)
    ∇Y = @. log(X) * X^Y 
    return Z, ∇Z -> (∇Z .* ∇X, ∇Z .* ∇Y)  
end
@⬅BinaryBroadcastedOL Base.:^

# =================== Sin
function ⬅Dual(::typeof(sin), x::Number)
    z = sin(x)
    ∂z∂x = cos(x)
    return z, ∂l∂z -> ∂l∂z*∂z∂x
end
@⬅UnaryScalarFunctionOL sin

# =================== Cos
function ⬅Dual(::typeof(cos), x::Number)
    z = cos(x)
    ∂z∂x = -sin(x)
    return z, ∂l∂z -> ∂l∂z*∂z∂x
end
@⬅UnaryScalarFunctionOL cos


function ⬅Dual(::typeof(ReLU), X::AbstractArray)
    Z = ReLU(X)
    return Z, ∇z -> ∇z .* map(x -> x > 0 ? x : 0, X)
end
@⬅UnaryFunctionOL ReLU 



function ⬅Dual(::typeof(Sigmoid), X::AbstractArray)
    Z = Sigmoid(X)
    return Z, ∇z -> ∇z .* map(x -> Sigmoid(x)*(1- Sigmoid(x)), X)
end
@⬅UnaryFunctionOL Sigmoid


function ⬅Dual(::typeof(map), f::Function, X::AbstractArray)
    # TODO
end


# ================================
# Linear Algebra
# ================================

function ⬅Dual(::typeof(*), X::AbstractArray, Y::AbstractArray)
    Z = X * Y
    return Z, ∇Z -> (∇Z * Y', X' * ∇Z)
end
@⬅BinaryFunctionOL Base.:*

function ⬅Dual(::typeof(adjoint), X::AbstractArray)
    return X', ∇Z -> ∇Z'
end
@⬅UnaryFunctionOL Base.adjoint



# ================================
# Restructuring
# ================================
function ⬅Dual(::typeof(getindex), X::T, indices...) where T<:Union{AbstractMatrix, AbstractVector}
    Z = getindex(X, indices...)
    function sparsefill(size, ∇z, indices...)
        S = spzeros(size)
        S[indices...] = ∇z
        return S
    end
    return Z, ∇z -> sparsefill(size(X), ∇z, indices...)
end
@⬅UnaryFunctionOL getindex


"""
function ⬅Dual(::typeof(sum), X::AbstractMatrix, dims)
    if dims == 1
        Z = sum(X, Dims=1)
        return Z, ∇z -> (∇z * Y')
    Z = sum(X, Dims=1)
    return Z, ∇z -> (∇z * Y')
end
"""
