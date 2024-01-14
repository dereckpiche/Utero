⮕Dualed = []

function ⮕Dual(f::typeof(+), a, b, ∇a, ∇b)
    return a+b, ∇a * ∇b
end
push!(⮕Dualed, :+)

function ⮕Dual(f::typeof(-), a, b, ∇a, ∇b)
    return a-b, ∇a - ∇b
end

