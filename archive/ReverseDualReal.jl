"""
    ⬅Dual(::typeof(f), x1::Number, x2::Number, ...)
"⬅" stands for "reverse mode"
On the left, return the result of the operation. 
With z = f(x1, x2, ...), on the right, return the 
'Chainer" function: ``∂l/∂z -> ∂l/∂x1, ∂l/∂x2, ...``
"""
function ⬅Dual(::typeof(+), x::Number, y::Number)
    z = x + y
    ∂z∂x = 1
    ∂z∂y = 1
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
@⬅BinaryScalarFunctionOL Base.:+

function ⬅Dual(::typeof(-), x::Number, y::Number)
    z = x - y
    ∂z∂x = 1
    ∂z∂y = -1
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
@⬅BinaryScalarFunctionOL Base.:-

function ⬅Dual(::typeof(*), x::Number, y::Number)
    z = x * y
    ∂z∂x = y
    ∂z∂y = x
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
@⬅BinaryScalarFunctionOL Base.:*


function ⬅Dual(::typeof(/), x::Number, y::Number)
    z = x / y
    ∂z∂x = y
    ∂z∂y = -x/y^2
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
@⬅BinaryScalarFunctionOL /


function ⬅Dual(::typeof(^), x::Number, y::Number)
    z = x^y
    ∂z∂x = y*x^(y-1)
    ∂z∂y = log(x) * x^y 
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
@⬅BinaryScalarFunctionOL ^


function ⬅Dual(::typeof(sin), x::Number)
    z = sin(x)
    ∂z∂x = cos(x)
    return z, ∂l∂z -> ∂l∂z*∂z∂x
end
@⬅UnaryScalarFunctionOL sin

function ⬅Dual(::typeof(cos), x::Number)
    z = cos(x)
    ∂z∂x = -sin(x)
    return z, ∂l∂z -> ∂l∂z*∂z∂x
end
@⬅UnaryScalarFunctionOL cos








