"""
    ⬅Dual(::typeof(f), x1, x2, ...)
On the left, return the result of the operation. 
With z = f(x1, x2, ...), on the right, return the function
``∂l/∂z -> ∂l/∂z, ∂l/∂x1, ∂l/∂x2, ...``
"""
function ⬅Dual(::typeof(+), x, y)
    z = x + y
    ∂z∂x = 1
    ∂z∂y = 1
    Linker(∂l∂z) = (∂l∂z, ∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
    return z, Linker
end

function ⬅Dual(::typeof(-), x, y)
    z = x - y
    ∂z∂x = 1
    ∂z∂y = -1
    Linker(∂l∂z) = (∂l∂z, ∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
    return z, Linker
end

function ⬅Dual(::typeof(*), x, y)
    z = x * y
    ∂z∂x = y
    ∂z∂y = x
    Linker(∂l∂z) = (∂l∂z, ∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
    return z, Linker
end

function ⬅Dual(::typeof(/), x, y)
    z = x * y
    ∂z∂x = y
    ∂z∂y = x
    Linker(∂l∂z) = (∂l∂z, ∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
    return z, Linker
end

function ⬅Dual(::typeof(^), x, y)
    z = x^y
    ∂z∂x = y*x^(y-1)
    ∂z∂y = log(x) * x^y 
    Linker(∂l∂z) = (∂l∂z, ∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
    return z, Linker
end

function ⬅Dual(::typeof(sin), x)
    z = sin(x)
    ∂z∂x = cos(x)
    Linker(∂l∂z) = (∂l∂z, ∂l∂z*∂z∂x)
    return z, Linker
end
function ⬅Dual(::typeof(cos), x)
    # TODO
    z = cos(x)
    ∂z∂x = -sin(x)
    return z, ∇z -> ∇z * cos(x)
end



