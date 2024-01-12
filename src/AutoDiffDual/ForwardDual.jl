
function ⮕Dual(f::typeof(+), a, b, ∇a, ∇b)
    return a+b, ∇a * ∇b
end

function ⮕Dual(f::typeof(-), a, b, ∇a, ∇b)
    return a-b, ∇a - ∇b
end

