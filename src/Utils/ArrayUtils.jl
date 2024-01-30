

GetOrder(X::AbstractArray) = length(size(X))

NbElements(x::AbstractArray) = prod(size(x))

function OneHot(loc::Int, size=10)
    Y = zeros(Float64, size, 1)
    Y[loc] = 1
    return Y
end