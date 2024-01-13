"""
    ⬅Dual(::typeof(f), x1, x2, ...)
On the left, return the result of the operation. 
With z = f(x1, x2, ...), on the right, return the function
``∂l/∂z -> ∂l/∂x1, ∂l/∂x2, ...``
"""
function ⬅Dual(::typeof(+), x, y)
    z = x + y
    ∂z∂x = 1
    ∂z∂y = 1
    AdditionLinker(∂l∂z) = (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
    return z, AdditionLinker
end

function ⬅Dual(::typeof(-), x, y)
    z = x - y
    ∂z∂x = 1
    ∂z∂y = -1
    SubstractionLinker(∂l∂z) = (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
    return z, SubstractionLinker
end

function ⬅Dual(::typeof(*), x, y)
    z = x * y
    ∂z∂x = y
    ∂z∂y = x
    MultiplicationLinker(∂l∂z) = (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
    return z, MultiplicationLinker
end

function ⬅Dual(::typeof(/), x, y)
    z = x * y
    ∂z∂x = y
    ∂z∂y = -x/y^2
    DivisionLinker(∂l∂z) = (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
    return z, DivisionLinker
end

function ⬅Dual(::typeof(^), x, y)
    z = x^y
    ∂z∂x = y*x^(y-1)
    ∂z∂y = log(x) * x^y 
    ExponentiationLinker(∂l∂z) = (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
    return z, ExponentiationLinker
end

function ⬅Dual(::typeof(sin), x::Real)
    z = sin(x)
    ∂z∂x = cos(x)
    SinLinker(∂l∂z) = (∂l∂z*∂z∂x)
    return z, SinLinker
end

function ⬅Dual(::typeof(cos), x::Real)
    z = cos(x)
    ∂z∂x = -sin(x)
    CosLinker(∂l∂z) = (∂l∂z*∂z∂x)
    return z, CosLinker
end



