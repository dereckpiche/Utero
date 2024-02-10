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

function ⬅Dual(::typeof(+), X, Y)
    Z = X + Y
    return Z, ∇z -> (∇z, ∇z)
end

# =================== Substraction 

function ⬅Dual(::typeof(-), x::Number, y::Number)
    z = x - y
    ∂z∂x = 1
    ∂z∂y = -1
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end

function ⬅Dual(::typeof(-), X, Y)
    Z = X - Y
    return Z, ∇z -> (∇z, -∇z)
end

# =================== Multiplication


function ⬅Dual(::typeof(*), x::Number, y::Number)
    z = x * y
    ∂z∂x = y
    ∂z∂y = x
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end

function ⬅Dual(::typeof(broadcasted), ::typeof(*), X, Y)
    Z = X .* Y
    # TODO: verify
    ∇X = ∇Z -> ∇Z .* Y
    ∇Y = ∇Z -> ∇Z .* X
    function BroadcastChainer(∇Z, ∇X, ∇Y)
        if NbElements(X) == NbElements(Y) return (∇X(∇Z), ∇Y(∇Z)) 
        elseif NbElements(X) > NbElements(Y) return (∇X(∇Z), sum(∇Y(∇Z))) end
        return (sum(∇X(∇Z)), ∇Y(∇Z))
    end
    return Z, ∇Z -> BroadcastChainer(∇Z, ∇X, ∇Y)
end

# =================== Divison

function ⬅Dual(::typeof(/), x::Number, y::Number)
    z = x / y
    ∂z∂x = y
    ∂z∂y = -x/y^2
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end

function ⬅Dual(::typeof(broadcasted), ::typeof(/), X, Y)
    Z = X ./ Y
    ∇X = ∇Z -> ∇Z ./ Y
    ∇Y = ∇Z -> ∇Z .* X
    function BroadcastChainer(∇Z, ∇X, ∇Y)
        if NbElements(X) == NbElements(Y) return (∇X(∇Z), ∇Y(∇Z)) 
        elseif NbElements(X) > NbElements(Y) return (∇X(∇Z), sum(∇Y(∇Z))) end
        return (sum(∇X(∇Z)), ∇Y(∇Z))
    end
    return Z, ∇Z -> BroadcastChainer(∇Z, ∇X, ∇Y)
end

# =================== Exponentiation

function ⬅Dual(::typeof(^), x::Number, y::Number)
    z = x^y
    ∂z∂x = y*x^(y-1)
    ∂z∂y = log(x) * x^y 
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end

function ⬅Dual(::typeof(broadcasted), ::typeof(exp), X)
    # TODO: fix
    Z = exp.(X)
    ∇X = ∇Z -> ∇Z .* Z
    function BroadcastChainer(∇Z, ∇X)
        (NbElements(X) > 1) ? (return sum(∇X(∇Z))) : return ∇X(∇Z)
    end
    return Z, ∇Z -> BroadcastChainer(∇Z, ∇X)
end

function ⬅Dual(::typeof(broadcasted), ::typeof(^), X, Y)
    Z = X .^ Y
    ∇X = ∇Z -> ∇Z .* X.^(Y-1)
    ∇Y = ∇Z -> ∇Z .* log(X) .* X .^ Y 
    function BroadcastChainer(∇Z, ∇X, ∇Y)
        if NbElements(X) == NbElements(Y) return (∇X(∇Z), ∇Y(∇Z)) 
        elseif NbElements(X) > NbElements(Y) return (∇X(∇Z), sum(∇Y(∇Z))) end
        return (sum(∇X(∇Z)), ∇Y(∇Z))
    end
    return Z, ∇Z -> BroadcastChainer(∇Z, ∇X, ∇Y)
end

# =================== Sin
function ⬅Dual(::typeof(sin), x::Number)
    z = sin(x)
    ∂z∂x = cos(x)
    return z, ∂l∂z -> ∂l∂z*∂z∂x
end

# =================== Cos
function ⬅Dual(::typeof(cos), x::Number)
    z = cos(x)
    ∂z∂x = -sin(x)
    return z, ∂l∂z -> ∂l∂z*∂z∂x
end


function ⬅Dual(::typeof(ReLU), X::AbstractArray)
    Z = ReLU(X)
    return Z, ∇z -> ∇z .* map(x -> x > 0 ? x : 0, X)
end


function ⬅Dual(::typeof(Sigmoid), X::AbstractArray)
    Z = Sigmoid(X)
    return Z, ∇z -> ∇z .* map(x -> Sigmoid(x)*(1- Sigmoid(x)), X)
end


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

function ⬅Dual(::typeof(adjoint), X)
    return X', ∇Z -> ∇Z'
end



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

function ⬅Dual(::typeof(sum), X::AbstractMatrix, dims)
    # TODO
    if dims == 1
        Z = sum(X, Dims=1)
        return Z, ∇z -> (∇z * Y')
    Z = sum(X, Dims=1)
    return Z, ∇z -> (∇z * Y')
end
