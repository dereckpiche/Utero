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
    X, Y = SameOrder(X, Y)
    Z = X .* Y
    BroadcastChainer = ∇Z -> begin
        ∇X = ∇Z .* Y
        ∇Y = ∇Z .* X
        sz, sx, sy = size(Z), size(X), size(Y)
        sz == sx ? nothing : ∇X = sum(∇X, dims=findall(sx .< sz))
        sz == sy ? nothing : ∇Y = sum(∇Y, dims=findall(sy .< sz))
        return ∇X, ∇Y
    end
    return Z, ∇Z -> BroadcastChainer(∇Z)
end

# =================== Divison

function ⬅Dual(::typeof(/), x::Number, y::Number)
    z = x / y
    ∂z∂x = 1 / y
    ∂z∂y = -x / y^2
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end

function ⬅Dual(::typeof(broadcasted), ::typeof(/), X, Y)
    X, Y = SameOrder(X, Y)
    Z = X ./ Y
    BroadcastChainer = ∇Z -> begin
        ∇X = ∇Z ./ Y
        ∇Y = ∇Z .* ( .- X ./ (Y .^ 2) )
        sz, sx, sy = size(Z), size(X), size(Y)
        sz == sx ? nothing : ∇X = sum(∇X, dims=findall(sx .< sz))
        sz == sy ? nothing : ∇Y = sum(∇Y, dims=findall(sy .< sz))
        return ∇X, ∇Y
    end
    return Z, ∇Z -> BroadcastChainer(∇Z)
end


# =================== Exponentiation

function ⬅Dual(::typeof(^), x::Number, y::Number)
    z = x^y
    ∂z∂x = y*x^(y-1)
    ∂z∂y = log(x) * x^y 
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end

function ⬅Dual(::typeof(broadcasted), ::typeof(exp), X)
    Z = exp.(X)
    ∇X = ∇Z -> ∇Z .* Z
    return Z, ∇Z -> ∇X(∇Z)
end

function ⬅Dual(::typeof(broadcasted), ::typeof(^), X, Y)
    X, Y = SameOrder(X, Y)
    Z = X .^ Y
    BroadcastChainer = ∇Z -> begin
        ∇X = ∇Z .* X.^(Y-1)
        ∇Y = ∇Z .* log(X) .* X .^ Y 
        sz, sx, sy = size(Z), size(X), size(Y)
        sz == sx ? nothing : ∇X = sum(∇X, dims=findall(sx .< sz))
        sz == sy ? nothing : ∇Y = sum(∇Y, dims=findall(sy .< sz))
        return ∇X, ∇Y
    end
    return Z, ∇Z -> BroadcastChainer(∇Z)
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
    return Z, ∇z -> ∇z .* map(x -> x > 0 ? 1 : 0, X)
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

# =================== Matrix Multiplication

function ⬅Dual(::typeof(*), X::AbstractArray, Y::AbstractArray)
    Z = X * Y
    return Z, ∇Z -> (∇Z * Y', X' * ∇Z)
end


# =================== Transposition

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


function ⬅Dual(::typeof(sum), X)
    Z = sum(X)
    return Z, ∇Z -> fill(∇Z, size(X))
end

function ⬅Dual(::typeof(sum), X; dims=1)
    Z = sum(X, dims=dims)
    dims = filter(dim -> !(dim in dims), 1:length(size(X)))
    IsaWrappedFloat(Z) ? CleanZ = Z : CleanZ = dropdims(Z)
    return Z, ∇Z -> mapslices(_ -> CleanZ, zeros(size(X)), dims=dims)
end
