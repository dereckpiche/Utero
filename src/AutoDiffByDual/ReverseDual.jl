⬅Dualed = []

"""
    ⬅Dual(::typeof(f), x1::Real, x2::Real, ...)
"⬅" stands for "reverse mode"
On the left, return the result of the operation. 
With z = f(x1, x2, ...), on the right, return the function
``∂l/∂z -> ∂l/∂x1, ∂l/∂x2, ...``
"""
function ⬅Dual(::typeof(+), x::Real, y::Real)
    z = x + y
    ∂z∂x = 1
    ∂z∂y = 1
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
push!(⬅Dualed, :+)

function ⬅Dual(::typeof(-), x::Real, y::Real)
    z = x - y
    ∂z∂x = 1
    ∂z∂y = -1
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
push!(⬅Dualed, :-)

function ⬅Dual(::typeof(*), x::Real, y::Real)
    z = x * y
    ∂z∂x = y
    ∂z∂y = x
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
push!(⬅Dualed, :*)

function ⬅Dual(::typeof(/), x::Real, y::Real)
    z = x * y
    ∂z∂x = y
    ∂z∂y = -x/y^2
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
push!(⬅Dualed, :/)

function ⬅Dual(::typeof(^), x::Real, y::Real)
    z = x^y
    ∂z∂x = y*x^(y-1)
    ∂z∂y = log(x) * x^y 
    return z, (∂l∂z) -> (∂l∂z*∂z∂x, ∂l∂z*∂z∂y)
end
push!(⬅Dualed, :^)

function ⬅Dual(::typeof(sin), x::Real)
    z = sin(x)
    ∂z∂x = cos(x)
    return z, ∂l∂z -> ∂l∂z*∂z∂x
end
push!(⬅Dualed, :sin)

function ⬅Dual(::typeof(cos), x::Real)
    z = cos(x)
    ∂z∂x = -sin(x)
    return z, ∂l∂z -> ∂l∂z*∂z∂x
end
push!(⬅Dualed, :cos)


