⮕Dualed = []
"""
    ⮕Dual(::typeof(f), x1, x2, ...)
"⮕" stands for "forward mode"
"""
function ⮕Dual(f::typeof(+), a, b, ∇a, ∇b)
    return a+b, ∇a * ∇b
end
push!(⮕Dualed, :+)

function ⮕Dual(f::typeof(-), a, b, ∇a, ∇b)
    return a-b, ∇a - ∇b
end
